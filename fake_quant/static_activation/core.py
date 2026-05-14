from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping

import torch
import torch.nn as nn

from ..quant import FP8_MAX
from ..smoothquant.core import _should_quantize_name


def static_tensor_scale_from_absmax(
    activation_absmax: torch.Tensor,
    *,
    fp8_max: float = FP8_MAX,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Convert calibrated activation absmax stats to one static FP8 scale."""
    scale = activation_absmax.detach().float().abs().amax() / fp8_max
    return torch.clamp(scale, min=eps).cpu()


def compute_static_tensor_activation_scales_for_model(
    model: nn.Module,
    activation_absmax: Mapping[str, torch.Tensor],
    *,
    skip_module_names: Iterable[str] = ("lm_head",),
    target_regex: str | None = None,
    skip_regex: str | None = None,
) -> Dict[str, torch.Tensor]:
    """Compute one static activation scale per quantized Linear module."""
    skip_names = set(skip_module_names)
    target_pattern = re.compile(target_regex) if target_regex else None
    skip_pattern = re.compile(skip_regex) if skip_regex else None

    scales: Dict[str, torch.Tensor] = {}
    missing: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not _should_quantize_name(
            name,
            name.rsplit(".", 1)[-1],
            skip_names,
            target_pattern,
            skip_pattern,
        ):
            continue
        stats = activation_absmax.get(name)
        if stats is None:
            missing.append(name)
            continue
        scales[name] = static_tensor_scale_from_absmax(stats)

    if missing:
        raise KeyError(
            "Missing static activation absmax stats for modules: "
            + ", ".join(missing[:8])
            + (" ..." if len(missing) > 8 else "")
        )
    return scales
