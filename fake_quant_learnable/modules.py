from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import (
    ActQuant,
    FP8_MAX,
    activation_per_token_qdq_forward,
    activation_per_token_qdq_forward_tail_protected,
    fp8_weight_per_channel_forward,
)


class BaselineFakeQuantLinear(nn.Module):
    """Inference-time min-max W8A8 fake-quant Linear wrapper."""

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

    def forward_prepared(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_qdq, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_quant == "per_token":
            x = activation_per_token_qdq_forward(x, qmax=self.qmax, eps=self.eps)
        return self.forward_prepared(x)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"act_quant={self.act_quant}, qmax={self.qmax}"
        )


class SmoothQuantFakeQuantLinear(nn.Module):
    """Fixed SmoothQuant W8A8 fake-quant Linear wrapper.

    If ``input_scale`` is present, forward uses x / scale before activation
    quantization. If SmoothQuant folding has already moved the scale into the
    previous module, ``input_scale`` is None and the wrapper behaves like a
    normal W8A8 wrapper using the already-smoothed weight.
    """

    def __init__(
        self,
        *,
        weight_qdq: torch.Tensor,
        bias: torch.Tensor | None,
        act_quant: ActQuant = "none",
        input_scale: torch.Tensor | None = None,
        qmax: float = FP8_MAX,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if act_quant not in ("none", "per_token"):
            raise ValueError(f"Unsupported act_quant: {act_quant}")
        if weight_qdq.ndim != 2:
            raise ValueError(f"Expected 2D Linear weight, got shape {tuple(weight_qdq.shape)}")

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
        if input_scale is None:
            self.register_buffer("input_scale", None, persistent=True)
        else:
            scale = input_scale.detach().float().reshape(-1)
            if scale.numel() != self.in_features:
                raise ValueError(
                    f"Expected input_scale shape ({self.in_features},), got {tuple(input_scale.shape)}"
                )
            self.register_buffer("input_scale", scale, persistent=True)

    def smooth_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_scale is None:
            return x
        view_shape = (1,) * (x.ndim - 1) + (-1,)
        return x / self.input_scale.to(device=x.device, dtype=x.dtype).view(view_shape)

    def forward_prepared(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_qdq, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.smooth_activation(x)
        if self.act_quant == "per_token":
            x = activation_per_token_qdq_forward(x, qmax=self.qmax, eps=self.eps)
        return self.forward_prepared(x)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"act_quant={self.act_quant}, folded={self.input_scale is None}, qmax={self.qmax}"
        )


class GPTQFakeQuantLinear(nn.Module):
    """Inference-time GPTQ-calibrated FP8 weight + optional FP8 activation wrapper."""

    def __init__(
        self,
        *,
        weight_qdq: torch.Tensor,
        bias: torch.Tensor | None,
        act_quant: ActQuant = "none",
        qmax: float = FP8_MAX,
        eps: float = 1e-12,
        activation_tail_tokens: int = 0,
    ) -> None:
        super().__init__()
        if act_quant not in ("none", "per_token"):
            raise ValueError(f"Unsupported act_quant: {act_quant}")
        if weight_qdq.ndim != 2:
            raise ValueError(f"Expected 2D Linear weight, got shape {tuple(weight_qdq.shape)}")

        self.in_features = int(weight_qdq.shape[1])
        self.out_features = int(weight_qdq.shape[0])
        self.act_quant = act_quant
        self.qmax = float(qmax)
        self.eps = float(eps)
        if activation_tail_tokens < 0:
            raise ValueError(f"activation_tail_tokens must be non-negative, got {activation_tail_tokens}")
        self.activation_tail_tokens = int(activation_tail_tokens)
        self.register_buffer("weight_qdq", weight_qdq.detach().clone(), persistent=True)
        if bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", bias.detach().clone(), persistent=True)

    def forward_prepared(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_qdq, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_quant == "per_token":
            if self.activation_tail_tokens > 0:
                x = activation_per_token_qdq_forward_tail_protected(
                    x,
                    tail_tokens=self.activation_tail_tokens,
                    qmax=self.qmax,
                    eps=self.eps,
                )
            else:
                x = activation_per_token_qdq_forward(x, qmax=self.qmax, eps=self.eps)
        return self.forward_prepared(x)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"act_quant={self.act_quant}, qmax={self.qmax}, "
            f"activation_tail_tokens={self.activation_tail_tokens}"
        )
