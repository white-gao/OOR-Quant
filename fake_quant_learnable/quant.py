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


def protect_tail_tokens(quantized: torch.Tensor, original: torch.Tensor, tail_tokens: int) -> torch.Tensor:
    """Restore the final sequence tokens after activation QDQ."""
    if tail_tokens <= 0 or original.ndim < 2:
        return quantized
    seq_len = int(original.shape[-2])
    if seq_len <= 0:
        return quantized
    tail = min(int(tail_tokens), seq_len)
    protected = quantized.clone()
    protected[..., -tail:, :] = original[..., -tail:, :]
    return protected


def activation_per_token_qdq_forward_tail_protected(
    x: torch.Tensor,
    *,
    tail_tokens: int,
    qmax: float = FP8_MAX,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-token activation QDQ with the final sequence tokens kept in FP dtype."""
    quantized = activation_per_token_qdq_forward(x, qmax=qmax, eps=eps)
    return protect_tail_tokens(quantized, x, tail_tokens)
