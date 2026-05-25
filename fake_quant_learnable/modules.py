from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import (
    ActQuant,
    FP8_MAX,
    activation_per_token_qdq_forward,
    activation_per_token_qdq_ste,
    fp8_weight_per_channel_forward,
    lwt_weight_qdq_ste,
)


class BaselineFakeQuantLinear(nn.Module):
    """Inference-time min-max W+A FP8 fake-quant Linear wrapper."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        act_quant: ActQuant = "none",
        qmax: float = FP8_MAX,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if act_quant not in ("none", "per_token"):
            raise ValueError(f"Unsupported act_quant: {act_quant}")

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.act_quant = act_quant
        self.qmax = float(qmax)
        self.eps = float(eps)

        with torch.no_grad():
            weight_qdq = fp8_weight_per_channel_forward(
                linear.weight.detach(),
                qmax=self.qmax,
                eps=self.eps,
            )
        self.register_buffer("weight_qdq", weight_qdq, persistent=True)
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone(), persistent=True)
        else:
            self.register_buffer("bias", None, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_quant == "per_token":
            x = activation_per_token_qdq_forward(x, qmax=self.qmax, eps=self.eps)
        return F.linear(x, self.weight_qdq, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"act_quant={self.act_quant}, qmax={self.qmax}"
        )


class LearnableFakeQuantLinear(nn.Module):
    """Calibration-time Linear wrapper for learnable PTQ.

    M1 learns per-output-channel clipping. M2 additionally learns a per-input
    channel equivalent transform: x' = x / s and W' = W * s. The FP8 QDQ
    forward path is torch.float8_e4m3fn; STE is only used for calibration
    gradients.
    """

    def __init__(
        self,
        linear: nn.Linear,
        *,
        act_quant: ActQuant = "none",
        init_clip_multiplier: float = 1.0,
        min_clip_multiplier: float = 0.05,
        max_clip_multiplier: float = 4.0,
        enable_let: bool = False,
        init_let_scale: float = 1.0,
        min_let_scale: float = 0.05,
        max_let_scale: float = 20.0,
        qmax: float = FP8_MAX,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if act_quant not in ("none", "per_token"):
            raise ValueError(f"Unsupported act_quant: {act_quant}")
        if init_clip_multiplier <= 0:
            raise ValueError("init_clip_multiplier must be positive.")
        if min_clip_multiplier <= 0 or max_clip_multiplier < min_clip_multiplier:
            raise ValueError("Invalid clip multiplier bounds.")
        if init_let_scale <= 0:
            raise ValueError("init_let_scale must be positive.")
        if min_let_scale <= 0 or max_let_scale < min_let_scale:
            raise ValueError("Invalid LET scale bounds.")

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.act_quant = act_quant
        self.min_clip_multiplier = float(min_clip_multiplier)
        self.max_clip_multiplier = float(max_clip_multiplier)
        self.enable_let = bool(enable_let)
        self.min_let_scale = float(min_let_scale)
        self.max_let_scale = float(max_let_scale)
        self.qmax = float(qmax)
        self.eps = float(eps)
        self.quant_enabled = True

        with torch.no_grad():
            weight = linear.weight.detach().clone()
            base_clip = torch.clamp(weight.float().abs().amax(dim=1, keepdim=True), min=eps)
        self.register_buffer("weight_frozen", weight, persistent=True)
        self.register_buffer("base_clip", base_clip.to(weight.device), persistent=True)
        if linear.bias is not None:
            self.register_buffer("bias_frozen", linear.bias.detach().clone(), persistent=True)
        else:
            self.register_buffer("bias_frozen", None, persistent=True)

        init_clip = torch.full_like(self.base_clip, float(init_clip_multiplier)).clamp(
            min=self.min_clip_multiplier,
            max=self.max_clip_multiplier,
        )
        self.log_clip_multiplier = nn.Parameter(torch.log(init_clip))

        if self.enable_let:
            init_let = torch.full(
                (self.in_features,),
                float(init_let_scale),
                dtype=torch.float32,
                device=weight.device,
            ).clamp(min=self.min_let_scale, max=self.max_let_scale)
            self.log_let_scale = nn.Parameter(torch.log(init_let))
        else:
            self.register_parameter("log_let_scale", None)

    @property
    def clip_multiplier(self) -> torch.Tensor:
        return torch.clamp(
            torch.exp(self.log_clip_multiplier),
            min=self.min_clip_multiplier,
            max=self.max_clip_multiplier,
        )

    @property
    def clip_base(self) -> torch.Tensor:
        if self.let_scale is None:
            return self.base_clip
        return (
            self.scaled_weight()
            .detach()
            .float()
            .abs()
            .amax(dim=1, keepdim=True)
            .clamp_min(self.eps)
            .to(device=self.weight_frozen.device)
        )

    @property
    def clip(self) -> torch.Tensor:
        return self.clip_base * self.clip_multiplier

    @property
    def let_scale(self) -> torch.Tensor | None:
        if self.log_let_scale is None:
            return None
        return torch.clamp(
            torch.exp(self.log_let_scale),
            min=self.min_let_scale,
            max=self.max_let_scale,
        )

    def set_quant_enabled(self, enabled: bool) -> None:
        self.quant_enabled = bool(enabled)

    def scaled_weight(self) -> torch.Tensor:
        scale = self.let_scale
        if scale is None:
            return self.weight_frozen
        scaled = self.weight_frozen.float() * scale.to(
            device=self.weight_frozen.device,
            dtype=torch.float32,
        ).view(1, -1)
        return scaled.to(dtype=self.weight_frozen.dtype)

    def quantized_weight(self) -> torch.Tensor:
        return lwt_weight_qdq_ste(
            self.scaled_weight(),
            self.clip.to(device=self.weight_frozen.device),
            qmax=self.qmax,
            eps=self.eps,
        )

    def _let_activation(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.let_scale
        if scale is None:
            return x
        view_shape = (1,) * (x.ndim - 1) + (-1,)
        return x / scale.to(device=x.device, dtype=x.dtype).view(view_shape)

    def _quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_quant == "per_token":
            return activation_per_token_qdq_ste(x, qmax=self.qmax, eps=self.eps)
        return x

    def forward_fp(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_frozen, self.bias_frozen)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quant_enabled:
            return self.forward_fp(x)
        x = self._let_activation(x)
        x = self._quantize_activation(x)
        return F.linear(x, self.quantized_weight(), self.bias_frozen)

    def to_frozen(self) -> "FrozenLearnedFakeQuantLinear":
        with torch.no_grad():
            weight_qdq = self.quantized_weight().detach().clone()
            bias = None if self.bias_frozen is None else self.bias_frozen.detach().clone()
            let_scale = None if self.let_scale is None else self.let_scale.detach().clone()
        return FrozenLearnedFakeQuantLinear(
            weight_qdq=weight_qdq,
            bias=bias,
            act_quant=self.act_quant,
            let_scale=let_scale,
            qmax=self.qmax,
            eps=self.eps,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"act_quant={self.act_quant}, enable_let={self.enable_let}, qmax={self.qmax}"
        )


class FrozenLearnedFakeQuantLinear(nn.Module):
    """Inference-time wrapper with learned weight QDQ and optional LET frozen."""

    def __init__(
        self,
        *,
        weight_qdq: torch.Tensor,
        bias: torch.Tensor | None,
        act_quant: ActQuant = "none",
        let_scale: torch.Tensor | None = None,
        qmax: float = FP8_MAX,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if act_quant not in ("none", "per_token"):
            raise ValueError(f"Unsupported act_quant: {act_quant}")
        self.in_features = int(weight_qdq.shape[1])
        self.out_features = int(weight_qdq.shape[0])
        self.act_quant = act_quant
        self.qmax = float(qmax)
        self.eps = float(eps)
        self.register_buffer("weight_qdq", weight_qdq.detach().clone(), persistent=True)
        if bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", bias.detach().clone(), persistent=True)
        if let_scale is None:
            self.register_buffer("let_scale", None, persistent=True)
        else:
            scale = let_scale.detach().float().reshape(-1)
            if scale.numel() != self.in_features:
                raise ValueError(
                    f"Expected let_scale shape ({self.in_features},), got {tuple(let_scale.shape)}"
                )
            self.register_buffer("let_scale", scale, persistent=True)

    def _let_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.let_scale is None:
            return x
        view_shape = (1,) * (x.ndim - 1) + (-1,)
        return x / self.let_scale.to(device=x.device, dtype=x.dtype).view(view_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._let_activation(x)
        if self.act_quant == "per_token":
            x = activation_per_token_qdq_forward(x, qmax=self.qmax, eps=self.eps)
        return F.linear(x, self.weight_qdq, self.bias)
