from __future__ import annotations

import math
import re
from typing import Iterable

import torch

LAYER_MODULE_PATTERN = re.compile(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)$")


def canonical_shared_input_name(module_name: str) -> str | None:
    match = LAYER_MODULE_PATTERN.search(module_name)
    if match is None:
        return None
    layer_idx, block_name, proj_name = match.groups()
    if block_name == "self_attn" and proj_name == "q_proj":
        return f"model.layers.{layer_idx}.attn_qkv_input"
    if block_name == "mlp" and proj_name == "gate_proj":
        return f"model.layers.{layer_idx}.ffn_gate_up_input"
    return None


def topk_count(num_channels: int, topk_fraction: float) -> int:
    if num_channels <= 0:
        raise ValueError("num_channels must be positive")
    if topk_fraction <= 0:
        raise ValueError("topk_fraction must be positive")
    return min(num_channels, max(1, int(math.ceil(num_channels * topk_fraction))))


def _topk_mask(scores: torch.Tensor, topk_fraction: float) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D [num_samples, num_channels], got {tuple(scores.shape)}")
    k = topk_count(scores.shape[1], topk_fraction)
    indices = torch.topk(scores.float(), k=k, dim=1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter_(1, indices, True)


def topk_selection_mask(scores: torch.Tensor, topk_fraction: float) -> torch.Tensor:
    return _topk_mask(scores, topk_fraction)


def topk_jaccard_matrix(scores: torch.Tensor, topk_fraction: float) -> torch.Tensor:
    mask = _topk_mask(scores, topk_fraction)
    intersection = (mask[:, None, :] & mask[None, :, :]).sum(dim=-1).float()
    union = (mask[:, None, :] | mask[None, :, :]).sum(dim=-1).float().clamp_min(1.0)
    return intersection / union


def topk_frequency(scores: torch.Tensor, topk_fraction: float) -> torch.Tensor:
    return _topk_mask(scores, topk_fraction).float().sum(dim=0)


def topk_overlap_stats(
    lhs_scores: torch.Tensor,
    rhs_scores: torch.Tensor,
    *,
    topk_fraction: float,
) -> dict[str, float | int]:
    if lhs_scores.shape != rhs_scores.shape:
        raise ValueError(
            "lhs_scores and rhs_scores must have the same shape, "
            f"got {tuple(lhs_scores.shape)} and {tuple(rhs_scores.shape)}"
        )
    lhs_mask = _topk_mask(lhs_scores, topk_fraction)
    rhs_mask = _topk_mask(rhs_scores, topk_fraction)
    intersection = (lhs_mask & rhs_mask).sum(dim=1).float()
    union = (lhs_mask | rhs_mask).sum(dim=1).float().clamp_min(1.0)
    return {
        "k": topk_count(lhs_scores.shape[1], topk_fraction),
        "jaccard_mean": float((intersection / union).mean().item()),
        "overlap_count_mean": float(intersection.mean().item()),
        "overlap_count_max": float(intersection.max().item()),
    }


def _offdiag_mean(matrix: torch.Tensor) -> float:
    if matrix.shape[0] <= 1:
        return 1.0
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return float(matrix[mask].mean().item())


def summarize_channel_stability(
    scores: torch.Tensor,
    *,
    topk_fractions: Iterable[float],
    eps: float = 1e-12,
) -> dict[str, float | int]:
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D [num_samples, num_channels], got {tuple(scores.shape)}")
    scores = scores.float()
    mean = scores.mean(dim=0)
    std = scores.std(dim=0, unbiased=False)
    cv = std / mean.abs().clamp_min(eps)

    summary: dict[str, float | int] = {
        "num_samples": int(scores.shape[0]),
        "num_channels": int(scores.shape[1]),
        "importance_mean": float(mean.mean().item()),
        "importance_max": float(mean.max().item()),
        "cv_mean": float(cv.mean().item()),
        "cv_median": float(cv.median().item()),
    }

    for fraction in topk_fractions:
        key = f"topk/{fraction:.6f}"
        jaccard = topk_jaccard_matrix(scores, fraction)
        frequency = topk_frequency(scores, fraction)
        summary[f"{key}/k"] = topk_count(scores.shape[1], fraction)
        summary[f"{key}/jaccard_mean"] = _offdiag_mean(jaccard)
        summary[f"{key}/frequency_max"] = float(frequency.max().item())
        summary[f"{key}/frequency_mean"] = float(frequency.mean().item())
    return summary
