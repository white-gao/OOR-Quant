from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quant import ActQuant, fp8_activation_per_token, fp8_weight_per_channel


@dataclass
class SmoothQuantSummary:
    replaced_linears: int
    skipped_linears: int
    shared_attention_modules: int = 0
    shared_mlp_modules: int = 0


def compute_smooth_scale(
    activation_absmax: torch.Tensor,
    weight_absmax: torch.Tensor,
    *,
    alpha: float = 0.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute SmoothQuant per-input-channel scales."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if activation_absmax.shape != weight_absmax.shape:
        raise ValueError(
            "activation_absmax and weight_absmax must have the same shape, "
            f"got {tuple(activation_absmax.shape)} and {tuple(weight_absmax.shape)}"
        )

    x = torch.clamp(activation_absmax.detach().float(), min=eps)
    w = torch.clamp(weight_absmax.detach().float(), min=eps)
    return torch.pow(x, alpha) / torch.pow(w, 1.0 - alpha)


def smooth_linear_weight(weight: torch.Tensor, smooth_scale: torch.Tensor) -> torch.Tensor:
    """Fold SmoothQuant input scales into Linear weight columns."""
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D Linear weight, got shape {tuple(weight.shape)}")
    if smooth_scale.ndim != 1 or smooth_scale.numel() != weight.shape[1]:
        raise ValueError(
            f"Expected scale shape ({weight.shape[1]},), got {tuple(smooth_scale.shape)}"
        )
    return weight * smooth_scale.to(device=weight.device, dtype=weight.dtype).view(1, -1)


class SmoothQuantLinear(nn.Module):
    """Linear with SmoothQuant input smoothing and cached FP8 weight QDQ."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        smooth_scale: torch.Tensor,
        act_quant: ActQuant = "none",
        smooth_input: bool = True,
    ) -> None:
        super().__init__()
        if act_quant not in ("none", "per_token"):
            raise ValueError(f"Unsupported act_quant: {act_quant}")

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.act_quant = act_quant
        self.smooth_input = smooth_input

        scale = smooth_scale.detach().float()
        if scale.ndim != 1 or scale.numel() != linear.in_features:
            raise ValueError(
                f"Expected smooth_scale shape ({linear.in_features},), got {tuple(scale.shape)}"
            )

        with torch.no_grad():
            scaled_weight = smooth_linear_weight(linear.weight.detach(), scale)
            weight_qdq = fp8_weight_per_channel(scaled_weight)

        self.register_buffer("smooth_scale", scale.to(linear.weight.device), persistent=True)
        self.register_buffer("weight_qdq", weight_qdq, persistent=True)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

    def smooth_activation(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.smooth_scale.to(device=x.device, dtype=x.dtype)
        return x / scale.view(*((1,) * (x.ndim - 1)), -1)

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_quant == "per_token":
            return fp8_activation_per_token(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.smooth_input:
            x = self.smooth_activation(x)
        x = self.quantize_activation(x)
        return F.linear(x, self.weight_qdq, self.bias)


def load_activation_absmax(path: str | Path) -> Dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, Mapping) and "x_absmax" in data:
        data = data["x_absmax"]
    if not isinstance(data, Mapping):
        raise ValueError(f"Unsupported SmoothQuant scale file format: {path}")
    return {str(name): tensor.detach().float().cpu() for name, tensor in data.items()}


def save_activation_absmax(
    path: str | Path,
    *,
    activation_absmax: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "oor_fake_quant_smoothquant_absmax_v1",
            "metadata": dict(metadata),
            "x_absmax": {
                name: tensor.detach().float().cpu() for name, tensor in activation_absmax.items()
            },
        },
        output_path,
    )


def compute_smooth_scales_for_model(
    model: nn.Module,
    activation_absmax: Mapping[str, torch.Tensor],
    *,
    alpha: float = 0.5,
    skip_module_names: Iterable[str] = ("lm_head",),
    target_regex: str | None = None,
    skip_regex: str | None = None,
) -> Dict[str, torch.Tensor]:
    """Compute per-Linear SmoothQuant scales, sharing qkv and gate/up scales."""
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
        scale = compute_smooth_scale(x_absmax, w_absmax, alpha=alpha)
        for name in available:
            scales[name] = scale
            used.add(name)

    for name, linear in linears.items():
        if name in used:
            continue
        x_absmax = _require_activation_stats(activation_absmax, [name])[0]
        w_absmax = _weight_input_absmax(linear)
        scales[name] = compute_smooth_scale(x_absmax, w_absmax, alpha=alpha)

    return scales


def _should_quantize_name(
    full_name: str,
    child_name: str,
    skip_names: set[str],
    target_pattern: re.Pattern[str] | None,
    skip_pattern: re.Pattern[str] | None,
) -> bool:
    if child_name in skip_names or full_name in skip_names:
        return False
    if skip_pattern is not None and skip_pattern.search(full_name) is not None:
        return False
    if target_pattern is not None and target_pattern.search(full_name) is None:
        return False
    return True


def _smoothquant_groups(linears: Mapping[str, nn.Linear]) -> list[tuple[str, ...]]:
    groups: list[tuple[str, ...]] = []
    prefixes: set[str] = set()
    for name in linears:
        if name.endswith(".self_attn.q_proj"):
            prefixes.add(name[: -len(".q_proj")])
        elif name.endswith(".mlp.gate_proj"):
            prefixes.add(name[: -len(".gate_proj")])

    for prefix in sorted(prefixes):
        if prefix.endswith(".self_attn"):
            groups.append((f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"))
        elif prefix.endswith(".mlp"):
            groups.append((f"{prefix}.gate_proj", f"{prefix}.up_proj"))
    return groups


def _require_activation_stats(
    activation_absmax: Mapping[str, torch.Tensor],
    names: Iterable[str],
) -> list[torch.Tensor]:
    tensors = []
    missing = []
    for name in names:
        tensor = activation_absmax.get(name)
        if tensor is None:
            missing.append(name)
            continue
        tensors.append(tensor.detach().float().cpu())
    if missing:
        raise KeyError(
            "Missing SmoothQuant activation absmax stats for modules: "
            + ", ".join(missing[:8])
            + (" ..." if len(missing) > 8 else "")
        )
    return tensors


def _weight_input_absmax(linear: nn.Linear) -> torch.Tensor:
    return linear.weight.detach().float().abs().amax(dim=0).cpu()


def _max_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    tensor_list = list(tensors)
    if not tensor_list:
        raise ValueError("Expected at least one tensor")
    return torch.stack(tensor_list, dim=0).amax(dim=0)
