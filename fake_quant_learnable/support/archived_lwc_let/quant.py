from __future__ import annotations

from typing import Literal

import torch


FP8_MAX = 448.0
ActQuant = Literal["none", "per_token"]
ActQuantMode = Literal["per_linear", "shared_input"]


def require_fp8() -> torch.dtype:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(
            "torch.float8_e4m3fn is required for FP8 fake quantization. "
            "Please use a PyTorch build with FP8 dtype support."
        )
    return torch.float8_e4m3fn


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round in forward and use identity gradient in backward."""
    return x + (torch.round(x) - x).detach()


def _uniform_symmetric_qdq_ste(
    x: torch.Tensor,
    scale: torch.Tensor,
    *,
    qmax: float = FP8_MAX,
    eps: float = 1e-12,
) -> torch.Tensor:
    orig_dtype = x.dtype
    scale_float = torch.clamp(scale.float(), min=eps)
    q = round_ste(x.float() / scale_float)
    q = torch.clamp(q, min=-qmax, max=qmax)
    return (q * scale_float).to(orig_dtype)


def fp8_e4m3_qdq_forward(
    x: torch.Tensor,
    scale: torch.Tensor,
    *,
    qmax: float = FP8_MAX,
    eps: float = 1e-12,
) -> torch.Tensor:
    """FP8 E4M3 quantize/dequantize forward path using explicit scales."""
    fp8_dtype = require_fp8()
    orig_dtype = x.dtype
    scale_float = torch.clamp(scale.float(), min=eps)
    q = torch.clamp(x.float() / scale_float, min=-qmax, max=qmax)
    q = q.to(fp8_dtype)
    return (q.float() * scale_float).to(orig_dtype)


def fp8_e4m3_qdq_ste(
    x: torch.Tensor,
    scale: torch.Tensor,
    *,
    qmax: float = FP8_MAX,
    eps: float = 1e-12,
) -> torch.Tensor:
    """FP8 E4M3 forward with a uniform STE proxy for backward gradients."""
    qdq_forward = fp8_e4m3_qdq_forward(x, scale, qmax=qmax, eps=eps)
    qdq_proxy = _uniform_symmetric_qdq_ste(x, scale, qmax=qmax, eps=eps)
    return qdq_proxy + (qdq_forward - qdq_proxy).detach()


def fp8_weight_per_channel_forward(
    weight: torch.Tensor,
    *,
    qmax: float = FP8_MAX,
    eps: float = 1e-12,
) -> torch.Tensor:
    """FP8 fake quantize Linear weights per output channel."""
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D Linear weight, got shape {tuple(weight.shape)}")
    absmax = weight.detach().float().abs().amax(dim=1, keepdim=True)
    scale = torch.clamp(absmax / qmax, min=eps)
    return fp8_e4m3_qdq_forward(weight, scale, qmax=qmax, eps=eps)


def activation_per_token_qdq_forward(
    x: torch.Tensor,
    *,
    qmax: float = FP8_MAX,
    eps: float = 1e-12,
) -> torch.Tensor:
    """FP8 fake quantize activations per token/row along the hidden dimension."""
    absmax = x.detach().float().abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(absmax / qmax, min=eps)
    return fp8_e4m3_qdq_forward(x, scale, qmax=qmax, eps=eps)


def activation_per_token_qdq_ste(
    x: torch.Tensor,
    *,
    qmax: float = FP8_MAX,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Activation FP8 E4M3 QDQ forward with identity STE gradient to activations."""
    absmax = x.detach().float().abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(absmax / qmax, min=eps)
    return fp8_e4m3_qdq_ste(x, scale, qmax=qmax, eps=eps)


def symmetric_clip(weight: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
    """Symmetric clip that routes gradients to clip for saturated weights."""
    return torch.where(weight > clip, clip, torch.where(weight < -clip, -clip, weight))


def lwt_weight_qdq_ste(
    weight: torch.Tensor,
    clip: torch.Tensor,
    *,
    qmax: float = FP8_MAX,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Apply learnable symmetric clipping and FP8 E4M3 STE QDQ to Linear weights."""
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D Linear weight, got shape {tuple(weight.shape)}")
    clip_float = torch.clamp(clip.float(), min=eps)
    weight_float = weight.float()
    clipped = symmetric_clip(weight_float, clip_float)
    scale = clip_float / qmax
    return fp8_e4m3_qdq_ste(clipped, scale, qmax=qmax, eps=eps).to(weight.dtype)
