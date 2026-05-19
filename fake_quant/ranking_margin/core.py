from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch
import torch.nn as nn

from ..smoothquant.core import (
    _max_tensors,
    _require_activation_stats,
    _should_quantize_name,
    _smoothquant_groups,
    _weight_input_absmax,
    compute_smooth_scale,
)


def normalize_importance(
    importance: torch.Tensor,
    *,
    clip_min: float = 0.25,
    clip_max: float = 4.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalize channel importance by geometric mean, then clip."""
    if clip_min <= 0 or clip_max <= 0 or clip_min > clip_max:
        raise ValueError(f"Invalid clip range: [{clip_min}, {clip_max}]")
    x = torch.clamp(importance.detach().float(), min=eps)
    geom_mean = torch.exp(torch.mean(torch.log(x)))
    normalized = x / torch.clamp(geom_mean, min=eps)
    return torch.clamp(normalized, min=clip_min, max=clip_max)


def compute_ranking_margin_smooth_scale(
    activation_absmax: torch.Tensor,
    weight_absmax: torch.Tensor,
    rank_importance: torch.Tensor,
    *,
    alpha: float = 0.5,
    beta: float = 0.25,
    clip_min: float = 0.25,
    clip_max: float = 4.0,
) -> torch.Tensor:
    """Compute SmoothQuant scale corrected by ranking-margin channel importance."""
    if activation_absmax.shape != rank_importance.shape:
        raise ValueError(
            "activation_absmax and rank_importance must have the same shape, "
            f"got {tuple(activation_absmax.shape)} and {tuple(rank_importance.shape)}"
        )
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")

    base_scale = compute_smooth_scale(activation_absmax, weight_absmax, alpha=alpha)
    if beta == 0:
        return base_scale
    normalized = normalize_importance(rank_importance, clip_min=clip_min, clip_max=clip_max)
    return base_scale * torch.pow(normalized.to(base_scale.device), beta)


def load_rank_importance(path: str | Path) -> Dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, Mapping) and "rank_importance" in data:
        data = data["rank_importance"]
    if not isinstance(data, Mapping):
        raise ValueError(f"Unsupported ranking importance file format: {path}")
    return {str(name): tensor.detach().float().cpu() for name, tensor in data.items()}


def save_rank_importance(
    path: str | Path,
    *,
    rank_importance: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "oor_fake_quant_ranking_margin_importance_v1",
            "metadata": dict(metadata),
            "rank_importance": {
                name: tensor.detach().float().cpu() for name, tensor in rank_importance.items()
            },
        },
        output_path,
    )


def compute_ranking_margin_smooth_scales_for_model(
    model: nn.Module,
    activation_absmax: Mapping[str, torch.Tensor],
    rank_importance: Mapping[str, torch.Tensor],
    *,
    alpha: float = 0.5,
    beta: float = 0.25,
    clip_min: float = 0.25,
    clip_max: float = 4.0,
    skip_module_names: Iterable[str] = ("lm_head",),
    target_regex: str | None = None,
    skip_regex: str | None = None,
    ranking_layer_min: int | None = None,
    ranking_layer_cutoff: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Compute group-wise ranking-margin SmoothQuant scales for Linear modules."""
    skip_names = set(skip_module_names)
    target_pattern = re.compile(target_regex) if target_regex else None
    skip_pattern = re.compile(skip_regex) if skip_regex else None
    linears = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and _should_quantize_name(
            name,
            name.rsplit(".", 1)[-1],
            skip_names,
            target_pattern,
            skip_pattern,
        )
    }

    scales: Dict[str, torch.Tensor] = {}
    used: set[str] = set()

    for group in _smoothquant_groups(linears):
        available = [name for name in group if name in linears]
        if not available:
            continue
        x_absmax = _max_tensors(_require_activation_stats(activation_absmax, available))
        w_absmax = _max_tensors([_weight_input_absmax(linears[name]) for name in available])
        if _uses_ranking_margin_scale(available, ranking_layer_min, ranking_layer_cutoff):
            importance = _max_tensors(_require_importance_stats(rank_importance, available))
            scale = compute_ranking_margin_smooth_scale(
                x_absmax,
                w_absmax,
                importance,
                alpha=alpha,
                beta=beta,
                clip_min=clip_min,
                clip_max=clip_max,
            )
        else:
            scale = compute_smooth_scale(x_absmax, w_absmax, alpha=alpha)
        for name in available:
            scales[name] = scale
            used.add(name)

    for name, linear in linears.items():
        if name in used:
            continue
        x_absmax = _require_activation_stats(activation_absmax, [name])[0]
        w_absmax = _weight_input_absmax(linear)
        if _uses_ranking_margin_scale([name], ranking_layer_min, ranking_layer_cutoff):
            importance = _require_importance_stats(rank_importance, [name])[0]
            scales[name] = compute_ranking_margin_smooth_scale(
                x_absmax,
                w_absmax,
                importance,
                alpha=alpha,
                beta=beta,
                clip_min=clip_min,
                clip_max=clip_max,
            )
        else:
            scales[name] = compute_smooth_scale(x_absmax, w_absmax, alpha=alpha)

    return scales


def _uses_ranking_margin_scale(
    names: Iterable[str],
    ranking_layer_min: int | None,
    ranking_layer_cutoff: int | None,
) -> bool:
    if ranking_layer_min is None and ranking_layer_cutoff is None:
        return True
    if ranking_layer_min is not None and ranking_layer_min < 0:
        raise ValueError(f"ranking_layer_min must be non-negative, got {ranking_layer_min}")
    if ranking_layer_cutoff is not None and ranking_layer_cutoff < 0:
        raise ValueError(f"ranking_layer_cutoff must be non-negative, got {ranking_layer_cutoff}")
    if (
        ranking_layer_min is not None
        and ranking_layer_cutoff is not None
        and ranking_layer_min > ranking_layer_cutoff
    ):
        raise ValueError(
            "ranking_layer_min must be <= ranking_layer_cutoff, "
            f"got {ranking_layer_min} > {ranking_layer_cutoff}"
        )
    layer_indices = [_extract_layer_index(name) for name in names]
    known_indices = [idx for idx in layer_indices if idx is not None]
    if not known_indices:
        return True
    layer_idx = max(known_indices)
    if ranking_layer_min is not None and layer_idx < ranking_layer_min:
        return False
    if ranking_layer_cutoff is not None and layer_idx >= ranking_layer_cutoff:
        return False
    return True


def _extract_layer_index(name: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    if match is None:
        return None
    return int(match.group(1))


def _require_importance_stats(
    rank_importance: Mapping[str, torch.Tensor],
    names: Iterable[str],
) -> list[torch.Tensor]:
    tensors = []
    missing = []
    for name in names:
        tensor = rank_importance.get(name)
        if tensor is None:
            missing.append(name)
            continue
        tensors.append(tensor.detach().float().cpu())
    if missing:
        raise KeyError(
            "Missing ranking-margin importance stats for modules: "
            + ", ".join(missing[:8])
            + (" ..." if len(missing) > 8 else "")
        )
    return tensors
