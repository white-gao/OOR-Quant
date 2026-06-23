from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


FP8_MAX = 448.0


@dataclass(frozen=True)
class FP8PreparedInput:
    x_fp8: torch.Tensor
    scale: torch.Tensor
    leading_shape: tuple[int, ...]


def require_fp8_runtime() -> torch.dtype:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is required for real FP8 W8A8 inference.")
    if not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("torch._scaled_mm is required for real FP8 W8A8 inference.")
    return torch.float8_e4m3fn


def _safe_scale(absmax: torch.Tensor, *, qmax: float, eps: float) -> torch.Tensor:
    return torch.clamp(absmax.float() / float(qmax), min=float(eps))


def quantize_fp8(x: torch.Tensor, scale: torch.Tensor, *, qmax: float) -> torch.Tensor:
    fp8_dtype = require_fp8_runtime()
    return torch.clamp(x.float() / scale.float(), -float(qmax), float(qmax)).to(fp8_dtype)


def weight_scale_per_output_channel(weight: torch.Tensor, *, qmax: float, eps: float) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D Linear weight, got shape {tuple(weight.shape)}")
    return _safe_scale(weight.detach().float().abs().amax(dim=1, keepdim=True), qmax=qmax, eps=eps)


def activation_scale_per_token(x_2d: torch.Tensor, *, qmax: float, eps: float) -> torch.Tensor:
    if x_2d.ndim != 2:
        raise ValueError(f"Expected 2D activation matrix, got shape {tuple(x_2d.shape)}")
    return _safe_scale(x_2d.detach().float().abs().amax(dim=1, keepdim=True), qmax=qmax, eps=eps)


class RealFP8Linear(nn.Module):
    """Naive W8A8 FP8 Linear using torch._scaled_mm.

    Weight is quantized once per output channel. Activation is dynamically
    quantized per token/row at every forward. ``prepare_input`` allows sibling
    projections with the same input, such as q/k/v or gate/up, to reuse one
    activation quantization result.
    """

    def __init__(
        self,
        *,
        weight_fp8_t: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        in_features: int,
        out_features: int,
        qmax: float = FP8_MAX,
        eps: float = 1e-12,
        output_dtype: torch.dtype = torch.bfloat16,
        use_fast_accum: bool = False,
    ) -> None:
        super().__init__()
        require_fp8_runtime()
        if weight_fp8_t.ndim != 2:
            raise ValueError(f"Expected 2D transposed weight, got shape {tuple(weight_fp8_t.shape)}")
        if tuple(weight_fp8_t.shape) != (int(in_features), int(out_features)):
            raise ValueError(
                "weight_fp8_t shape must be "
                f"({int(in_features)}, {int(out_features)}), got {tuple(weight_fp8_t.shape)}"
            )
        if weight_scale.shape != (1, int(out_features)):
            raise ValueError(f"Expected weight_scale shape (1, {out_features}), got {tuple(weight_scale.shape)}")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.qmax = float(qmax)
        self.eps = float(eps)
        self.output_dtype = output_dtype
        self.use_fast_accum = bool(use_fast_accum)

        self.register_buffer("weight_fp8_t", weight_fp8_t.detach(), persistent=True)
        self.register_buffer("weight_scale", weight_scale.detach().float(), persistent=True)
        if bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", bias.detach().clone(), persistent=True)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        qmax: float = FP8_MAX,
        eps: float = 1e-12,
        output_dtype: torch.dtype | None = None,
        use_fast_accum: bool = False,
    ) -> "RealFP8Linear":
        if linear.weight.ndim != 2:
            raise ValueError(f"Expected 2D Linear weight, got shape {tuple(linear.weight.shape)}")
        out_features, in_features = linear.weight.shape
        chosen_output_dtype = output_dtype
        if chosen_output_dtype is None:
            chosen_output_dtype = linear.weight.dtype
            if chosen_output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
                chosen_output_dtype = torch.bfloat16

        weight = linear.weight.detach()
        scale = weight_scale_per_output_channel(weight, qmax=qmax, eps=eps)
        weight_fp8 = quantize_fp8(weight, scale, qmax=qmax)
        return cls(
            weight_fp8_t=weight_fp8.t(),
            weight_scale=scale.t().contiguous(),
            bias=linear.bias,
            in_features=int(in_features),
            out_features=int(out_features),
            qmax=qmax,
            eps=eps,
            output_dtype=chosen_output_dtype,
            use_fast_accum=use_fast_accum,
        )

    def _flatten_input(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input last dim {self.in_features}, got {tuple(x.shape)}")
        return x.reshape(-1, self.in_features).contiguous(), tuple(x.shape[:-1])

    def prepare_input(self, x: torch.Tensor) -> FP8PreparedInput:
        x_2d, leading_shape = self._flatten_input(x)
        act_scale = activation_scale_per_token(x_2d, qmax=self.qmax, eps=self.eps)
        x_fp8 = quantize_fp8(x_2d, act_scale, qmax=self.qmax)
        return FP8PreparedInput(x_fp8=x_fp8, scale=act_scale, leading_shape=leading_shape)

    def forward_prepared(self, prepared: FP8PreparedInput) -> torch.Tensor:
        if prepared.x_fp8.ndim != 2 or prepared.x_fp8.shape[1] != self.in_features:
            raise ValueError(
                f"Prepared input must have shape [M, {self.in_features}], got {tuple(prepared.x_fp8.shape)}"
            )
        if prepared.x_fp8.device.type == "cuda":
            y_2d = torch._scaled_mm(
                prepared.x_fp8,
                self.weight_fp8_t,
                scale_a=prepared.scale,
                scale_b=self.weight_scale,
                out_dtype=self.output_dtype,
                use_fast_accum=self.use_fast_accum,
            )
        else:
            x_qdq = prepared.x_fp8.float() * prepared.scale.float()
            weight_qdq_t = self.weight_fp8_t.float() * self.weight_scale.float()
            y_2d = (x_qdq @ weight_qdq_t).to(self.output_dtype)
        if self.bias is not None:
            y_2d = y_2d + self.bias.to(device=y_2d.device, dtype=y_2d.dtype)
        return y_2d.reshape(*prepared.leading_shape, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_prepared(self.prepare_input(x))

    def reference_qdq_forward(self, x: torch.Tensor) -> torch.Tensor:
        prepared = self.prepare_input(x)
        x_qdq = prepared.x_fp8.float() * prepared.scale.float()
        weight_qdq_t = self.weight_fp8_t.float() * self.weight_scale.float()
        y_2d = x_qdq @ weight_qdq_t
        if self.bias is not None:
            y_2d = y_2d + self.bias.to(device=y_2d.device, dtype=y_2d.dtype)
        return y_2d.to(self.output_dtype).reshape(*prepared.leading_shape, self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"weight_scale=per_output_channel, act_scale=per_token, "
            f"output_dtype={self.output_dtype}, use_fast_accum={self.use_fast_accum}"
        )

    def _apply(self, fn: Any, recurse: bool = True) -> "RealFP8Linear":
        super()._apply(fn, recurse=recurse)
        return self
