from __future__ import annotations

from typing import Literal

import torch


FP8_MAX = 448.0

def require_fp8() -> torch.dtype:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(
            "torch.float8_e4m3fn is required for FP8 fake quantization. "
            "Please use a PyTorch build with FP8 dtype support."
        )
    return torch.float8_e4m3fn


def fp8_fake_quant(
    x: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: float = FP8_MAX,
) -> torch.Tensor:
    """Quantize/dequantize with FP8 E4M3 using explicit scales."""
    fp8_dtype = require_fp8()
    orig_dtype = x.dtype
    q = torch.clamp(x.float() / scale.float(), min=-fp8_max, max=fp8_max)
    q = q.to(fp8_dtype)
    return (q.float() * scale.float()).to(orig_dtype)


def fp8_weight_per_channel(
    weight: torch.Tensor,
    fp8_max: float = FP8_MAX,
    eps: float = 1e-12,
) -> torch.Tensor:
    """FP8 fake quantize Linear weights per output channel."""
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D Linear weight, got shape {tuple(weight.shape)}")
    absmax = weight.detach().float().abs().amax(dim=1, keepdim=True)
    scale = torch.clamp(absmax / fp8_max, min=eps)
    return fp8_fake_quant(weight, scale, fp8_max=fp8_max)


def fp8_activation_per_token(
    x: torch.Tensor,
    fp8_max: float = FP8_MAX,
    eps: float = 1e-12,
) -> torch.Tensor:
    """FP8 fake quantize activations per token/row along the hidden dimension."""
    absmax = x.detach().float().abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(absmax / fp8_max, min=eps)
    return fp8_fake_quant(x, scale, fp8_max=fp8_max)


def fp8_activation_static_tensor(
    x: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: float = FP8_MAX,
) -> torch.Tensor:
    """FP8 fake quantize activations with one calibrated tensor scale."""
    return fp8_fake_quant(x, scale.to(device=x.device), fp8_max=fp8_max)


ActQuant = Literal["none", "per_token", "static_tensor"]
ActQuantMode = Literal["per_linear", "shared_input"]
