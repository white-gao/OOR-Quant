from __future__ import annotations

import gc
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from fake_quant_learnable.support.runtime_utils import _module_device, _move_tree_to_device
from fake_quant_learnable.support.smoothquant_runtime import Batch, _batch_to_args_kwargs
from fake_quant_learnable.token_weights import SLOT_TOKEN_GROUPS

from .apply import NaiveW8A8Summary, install_shared_input_activation_quantization
from .gptq_runtime import advance_layer_input_batches, capture_layer_input_batches, get_transformer_layers
from .modules import ActivationQuantMode, RealFP8Linear


EPS = 1e-12


@dataclass(frozen=True)
class ConditionalGPTQLinearStats:
    """Per-Linear statistics for hard slot-conditional GPTQ routing."""

    group_hessians: torch.Tensor
    global_hessian: torch.Tensor
    row_group_ids: torch.Tensor
    mean_entropy: float
    row_max_pi: torch.Tensor
    dominant_fraction: float


def conditional_hessian_enabled(
    stats: ConditionalGPTQLinearStats,
    *,
    max_entropy: float,
    min_dominant_probability: float,
    min_dominant_fraction: float,
) -> bool:
    return (
        stats.mean_entropy <= float(max_entropy)
        and float((stats.row_max_pi >= float(min_dominant_probability)).float().mean().item())
        >= float(min_dominant_fraction)
    )


def collect_conditional_gptq_stats(
    module: nn.Module,
    batches: Sequence[Batch],
    *,
    token_group_batches: Sequence[torch.Tensor],
    target_names: Iterable[str] | None = None,
    eps: float = EPS,
) -> dict[str, ConditionalGPTQLinearStats]:
    """Collect group Hessians and output-channel slot mixtures for one layer.

    The input side yields ``H_g = mean_{t in g}(x_t x_t^T)``. The Linear
    output side yields a per-row slot mixture from token-RMS-normalized output
    energy. No model parameters are changed in this function.
    """
    if len(batches) != len(token_group_batches):
        raise ValueError(
            f"token_group_batches length {len(token_group_batches)} does not match batches length {len(batches)}"
        )
    linears = _named_linear_modules(module)
    if target_names is not None:
        selected = set(target_names)
        linears = {name: linear for name, linear in linears.items() if name in selected}
    if not linears:
        return {}

    h_sums: dict[str, torch.Tensor] = {}
    output_energy_sums: dict[str, torch.Tensor] = {}
    group_counts: dict[str, torch.Tensor] = {}
    current_groups: torch.Tensor | None = None

    def make_hook(name: str, linear: nn.Linear):
        def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            nonlocal current_groups
            x = args[0] if args else kwargs.get("input", kwargs.get("hidden_states"))
            y = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(x) or not torch.is_tensor(y) or current_groups is None:
                return
            if x.shape[-1] != linear.in_features or y.shape[-1] != linear.out_features:
                raise ValueError(f"Unexpected Linear shape while collecting conditional stats for {name!r}.")
            x2d = x.detach().float().reshape(-1, linear.in_features)
            y2d = y.detach().float().reshape(-1, linear.out_features)
            groups = _flatten_slot_groups(current_groups, x=x, rows=x2d.shape[0]).to(device=x2d.device)
            if y2d.shape[0] != x2d.shape[0]:
                raise ValueError(f"Linear input/output token rows differ for {name!r}.")

            if name not in h_sums:
                h_sums[name] = torch.zeros(
                    len(SLOT_TOKEN_GROUPS), linear.in_features, linear.in_features, device=x2d.device, dtype=torch.float32
                )
                output_energy_sums[name] = torch.zeros(
                    len(SLOT_TOKEN_GROUPS), linear.out_features, device=x2d.device, dtype=torch.float32
                )
                group_counts[name] = torch.zeros(len(SLOT_TOKEN_GROUPS), device=x2d.device, dtype=torch.float32)

            normalized_y_sq = y2d.square() / y2d.square().mean(dim=1, keepdim=True).clamp_min(eps)
            for group_id in range(len(SLOT_TOKEN_GROUPS)):
                mask = groups == group_id
                if not bool(mask.any()):
                    continue
                x_group = x2d[mask]
                h_sums[name][group_id].add_(x_group.t().matmul(x_group))
                output_energy_sums[name][group_id].add_(normalized_y_sq[mask].sum(dim=0))
                group_counts[name][group_id].add_(float(mask.sum().item()))

        return hook

    handles = [
        linear.register_forward_hook(make_hook(name, linear), with_kwargs=True)
        for name, linear in linears.items()
    ]
    was_training = module.training
    target_device = _module_device(module)
    module.eval()
    try:
        with torch.no_grad():
            for batch, groups in zip(batches, token_group_batches):
                current_groups = groups.to(target_device)
                args, kwargs = _batch_to_args_kwargs(batch)
                module(*_move_tree_to_device(args, target_device), **_move_tree_to_device(kwargs, target_device))
    finally:
        for handle in handles:
            handle.remove()
        module.train(was_training)

    result: dict[str, ConditionalGPTQLinearStats] = {}
    for name, linear in linears.items():
        if name not in h_sums:
            identity = torch.eye(linear.in_features, dtype=torch.float32)
            result[name] = ConditionalGPTQLinearStats(
                group_hessians=identity.unsqueeze(0).repeat(len(SLOT_TOKEN_GROUPS), 1, 1) * eps,
                global_hessian=identity * eps,
                row_group_ids=torch.zeros(linear.out_features, dtype=torch.long),
                mean_entropy=1.0,
                row_max_pi=torch.full((linear.out_features,), 1.0 / float(len(SLOT_TOKEN_GROUPS))),
                dominant_fraction=0.0,
            )
            continue
        counts = group_counts[name]
        hessian_groups = h_sums[name] / counts.clamp_min(1.0).view(-1, 1, 1)
        missing = counts <= 0
        if bool(missing.any()):
            identity = torch.eye(linear.in_features, device=hessian_groups.device, dtype=hessian_groups.dtype)
            hessian_groups[missing] = identity * eps
        hessian_groups = 0.5 * (hessian_groups + hessian_groups.transpose(1, 2))
        global_hessian = h_sums[name].sum(dim=0) / counts.sum().clamp_min(1.0)
        global_hessian = 0.5 * (global_hessian + global_hessian.t())

        energy = output_energy_sums[name] / counts.clamp_min(1.0).unsqueeze(1)
        pi = energy.transpose(0, 1) / energy.sum(dim=0).clamp_min(eps).unsqueeze(1)
        empty_rows = energy.sum(dim=0) <= eps
        if bool(empty_rows.any()):
            pi[empty_rows] = 1.0 / float(len(SLOT_TOKEN_GROUPS))
        max_pi, row_group_ids = pi.max(dim=1)
        entropy = -(pi.clamp_min(eps) * pi.clamp_min(eps).log()).sum(dim=1) / math.log(len(SLOT_TOKEN_GROUPS))
        result[name] = ConditionalGPTQLinearStats(
            group_hessians=hessian_groups.detach().cpu(),
            global_hessian=global_hessian.detach().cpu(),
            row_group_ids=row_group_ids.detach().cpu(),
            mean_entropy=float(entropy.mean().item()),
            row_max_pi=max_pi.detach().cpu(),
            dominant_fraction=0.0,  # Filled by the caller's chosen probability threshold.
        )
        result[name] = _with_dominant_fraction(result[name], max_pi.detach().cpu(), threshold=0.35)
    return result


def _with_dominant_fraction(
    stats: ConditionalGPTQLinearStats,
    max_pi: torch.Tensor,
    *,
    threshold: float,
) -> ConditionalGPTQLinearStats:
    return ConditionalGPTQLinearStats(
        group_hessians=stats.group_hessians,
        global_hessian=stats.global_hessian,
        row_group_ids=stats.row_group_ids,
        mean_entropy=stats.mean_entropy,
        row_max_pi=stats.row_max_pi,
        dominant_fraction=float((max_pi >= threshold).float().mean().item()),
    )


def apply_conditional_gptq_real_w8a8_layers(
    *,
    model: nn.Module,
    model_batches: Sequence[Mapping[str, Any]],
    token_group_batches: Sequence[torch.Tensor],
    layer_indices: Sequence[int],
    output_dtype: torch.dtype,
    target_regex: str | None,
    skip_regex: str | None,
    use_fast_accum: bool,
    activation_quant_mode: ActivationQuantMode,
    decode_a16_when_single_token: bool,
    activation_tail_tokens: int,
    damp_percent: float,
    block_size: int,
    max_entropy: float,
    min_dominant_probability: float,
    min_dominant_fraction: float,
) -> NaiveW8A8Summary:
    """Apply hard conditional GPTQ to gated Linears and plain GPTQ elsewhere."""
    _validate_conditional_thresholds(
        max_entropy=max_entropy,
        min_dominant_probability=min_dominant_probability,
        min_dominant_fraction=min_dominant_fraction,
    )
    layers = get_transformer_layers(model)
    fp_inputs: list[Batch] | None = None
    stream_layer_idx: int | None = None
    replaced = 0
    skipped = 0
    for layer_idx in sorted(layer_indices):
        if fp_inputs is None:
            fp_inputs = capture_layer_input_batches(model=model, layer=layers[layer_idx], model_batches=model_batches)
            stream_layer_idx = layer_idx
        else:
            assert stream_layer_idx is not None
            while stream_layer_idx < layer_idx:
                fp_inputs = advance_layer_input_batches(layer=layers[stream_layer_idx], batches=fp_inputs)
                stream_layer_idx += 1

        layer = layers[layer_idx]
        initial_linears = _named_linear_modules(layer)
        target_names = [
            name for name in initial_linears if not _should_skip(name, target_regex=target_regex, skip_regex=skip_regex)
        ]
        stats = collect_conditional_gptq_stats(
            layer,
            fp_inputs,
            token_group_batches=token_group_batches,
            target_names=target_names,
        )
        next_fp_inputs = advance_layer_input_batches(layer=layer, batches=fp_inputs)
        enabled_names: list[str] = []
        for name, linear in initial_linears.items():
            if _should_skip(name, target_regex=target_regex, skip_regex=skip_regex):
                skipped += 1
                continue
            stat = stats[name]
            enabled = conditional_hessian_enabled(
                stat,
                max_entropy=max_entropy,
                min_dominant_probability=min_dominant_probability,
                min_dominant_fraction=min_dominant_fraction,
            )
            if enabled:
                quantized = RealFP8Linear.from_conditional_gptq_linear(
                    linear,
                    stat.group_hessians,
                    stat.row_group_ids,
                    output_dtype=output_dtype,
                    use_fast_accum=use_fast_accum,
                    activation_quant_mode=activation_quant_mode,
                    decode_a16_when_single_token=decode_a16_when_single_token,
                    activation_tail_tokens=activation_tail_tokens,
                    damp_percent=damp_percent,
                    block_size=block_size,
                )
                enabled_names.append(name)
            else:
                quantized = RealFP8Linear.from_gptq_linear(
                    linear,
                    stat.global_hessian,
                    output_dtype=output_dtype,
                    use_fast_accum=use_fast_accum,
                    activation_quant_mode=activation_quant_mode,
                    decode_a16_when_single_token=decode_a16_when_single_token,
                    activation_tail_tokens=activation_tail_tokens,
                    damp_percent=damp_percent,
                    block_size=block_size,
                )
            _set_module_by_name(layer, name, quantized)
            replaced += 1
        fp_inputs = next_fp_inputs
        stream_layer_idx = layer_idx + 1
        print(
            f"[hf_naive_w8a8] conditional_gptq layer={layer_idx} replaced_linears={len(target_names)}, "
            f"conditional_linears={enabled_names}"
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    shared_attention, shared_mlp = install_shared_input_activation_quantization(model)
    return NaiveW8A8Summary(
        replaced_linears=replaced,
        skipped_linears=skipped,
        shared_attention_modules=shared_attention,
        shared_mlp_modules=shared_mlp,
    )


def _dominant_fraction(stats: ConditionalGPTQLinearStats, *, min_dominant_probability: float) -> float:
    # Reconstruct the exact fraction from the stored row assignments is not possible;
    # the collector currently stores it at the default probe threshold only.
    # The minimal implementation requires the same threshold at collection and gating.
    del min_dominant_probability
    return stats.dominant_fraction


def _validate_conditional_thresholds(*, max_entropy: float, min_dominant_probability: float, min_dominant_fraction: float) -> None:
    if not 0.0 <= max_entropy <= 1.0:
        raise ValueError(f"max_entropy must be in [0, 1], got {max_entropy}")
    if not 0.0 <= min_dominant_probability <= 1.0:
        raise ValueError(f"min_dominant_probability must be in [0, 1], got {min_dominant_probability}")
    if not 0.0 <= min_dominant_fraction <= 1.0:
        raise ValueError(f"min_dominant_fraction must be in [0, 1], got {min_dominant_fraction}")


def _flatten_slot_groups(groups: torch.Tensor, *, x: torch.Tensor, rows: int) -> torch.Tensor:
    if tuple(groups.shape) == tuple(x.shape[:-1]):
        return groups.reshape(-1)
    if groups.numel() == rows:
        return groups.reshape(-1)
    raise ValueError(f"Slot-group shape {tuple(groups.shape)} is incompatible with Linear input shape {tuple(x.shape)}.")


def _named_linear_modules(module: nn.Module) -> dict[str, nn.Linear]:
    return {name: child for name, child in module.named_modules() if isinstance(child, nn.Linear)}


def _should_skip(name: str, *, target_regex: str | None, skip_regex: str | None) -> bool:
    if name == "lm_head" or name.rsplit(".", 1)[-1] == "lm_head":
        return True
    if skip_regex and re.search(skip_regex, name):
        return True
    return bool(target_regex and re.search(target_regex, name) is None)


def _set_module_by_name(module: nn.Module, name: str, replacement: nn.Module) -> None:
    parent = module
    pieces = name.split(".")
    for piece in pieces[:-1]:
        parent = getattr(parent, piece)
    setattr(parent, pieces[-1], replacement)
