from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gradient_weights import (
    DEFAULT_GRADIENT_TOKEN_WEIGHT_CONFIG,
    GradientTokenWeightConfig,
    _build_teacher_forcing_batch,
    _collapse_teacher_forcing_prompt_sensitivity,
    _extract_logits,
    _forward_model,
    _last_prompt_positions,
    _target_tensor,
    _teacher_forcing_full_sid_loss,
    normalize_gradient_token_weights,
)

DEFAULT_LINEAR_SLOT_WEIGHT_REGEX = r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def collect_linear_gradient_group_token_weight_batches_by_layer(
    *,
    model: nn.Module,
    layers: Sequence[nn.Module],
    layer_indices: Sequence[int],
    model_batches: Sequence[Mapping[str, Any]],
    token_group_batches: Sequence[torch.Tensor],
    target_token_ids: Sequence[torch.Tensor | int] | None = None,
    teacher_forcing_target_token_ids: Sequence[Sequence[torch.Tensor | int]] | None = None,
    linear_regex: str = DEFAULT_LINEAR_SLOT_WEIGHT_REGEX,
    config: GradientTokenWeightConfig = DEFAULT_GRADIENT_TOKEN_WEIGHT_CONFIG,
) -> dict[int, dict[str, list[torch.Tensor]]]:
    """Collect Linear-wise, group-smoothed output-gradient token weights."""
    selected = sorted(set(int(idx) for idx in layer_indices))
    if not selected:
        return {}
    use_teacher_forcing = teacher_forcing_target_token_ids is not None
    if (target_token_ids is None) == (teacher_forcing_target_token_ids is None):
        raise ValueError("Pass exactly one of target_token_ids or teacher_forcing_target_token_ids.")
    expected_targets = teacher_forcing_target_token_ids if use_teacher_forcing else target_token_ids
    if expected_targets is None:
        raise RuntimeError("Internal error: gradient targets are not initialized.")
    if len(model_batches) != len(expected_targets):
        raise ValueError("gradient target length does not match model_batches length.")
    if len(model_batches) != len(token_group_batches):
        raise ValueError("token_group_batches length does not match model_batches length.")
    for layer_idx in selected:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of range for {len(layers)} layers.")

    pattern = re.compile(linear_regex)
    min_selected = selected[0]
    captured: dict[tuple[int, str], torch.Tensor] = {}
    handles = []
    raw_weights: dict[int, dict[str, list[torch.Tensor]]] = {layer_idx: {} for layer_idx in selected}

    def make_seed_hook(layer_idx: int):
        def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
            if layer_idx != min_selected:
                return None
            if args:
                hidden = args[0]
                if not torch.is_tensor(hidden):
                    return None
                return (hidden.detach().requires_grad_(True), *args[1:]), kwargs
            key = "hidden_states" if "hidden_states" in kwargs else "input"
            hidden = kwargs.get(key)
            if not torch.is_tensor(hidden):
                return None
            new_kwargs = dict(kwargs)
            new_kwargs[key] = hidden.detach().requires_grad_(True)
            return args, new_kwargs
        return hook

    def make_linear_hook(layer_idx: int, module_name: str):
        def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            if torch.is_tensor(output) and output.requires_grad:
                output.retain_grad()
                captured[(layer_idx, module_name)] = output
        return hook

    handles.append(layers[min_selected].register_forward_pre_hook(make_seed_hook(min_selected), with_kwargs=True))
    for layer_idx in selected:
        for module_name, module in layers[layer_idx].named_modules():
            if isinstance(module, nn.Linear) and pattern.search(module_name):
                raw_weights[layer_idx][module_name] = []
                handles.append(module.register_forward_hook(make_linear_hook(layer_idx, module_name)))

    was_training = model.training
    parameter_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    try:
        for batch, target_info in zip(model_batches, expected_targets):
            captured.clear()
            model.zero_grad(set_to_none=True)
            attention_mask = batch.get("attention_mask") if isinstance(batch, Mapping) else None
            if use_teacher_forcing:
                expanded_batch, prompt_mask, prompt_len, loss_specs = _build_teacher_forcing_batch(
                    batch, target_info  # type: ignore[arg-type]
                )
                logits = _extract_logits(_forward_model(model, expanded_batch))
                loss = _teacher_forcing_full_sid_loss(logits, loss_specs)
            else:
                logits = _extract_logits(_forward_model(model, batch))
                target = _target_tensor(target_info, logits=logits)  # type: ignore[arg-type]
                positions = _last_prompt_positions(batch, logits=logits)
                batch_indices = torch.arange(logits.shape[0], device=logits.device)
                loss = F.cross_entropy(logits[batch_indices, positions, :].float(), target)
                prompt_mask = None
                prompt_len = None
            loss.backward()

            for (layer_idx, module_name), output in captured.items():
                if output.grad is None:
                    continue
                sensitivity = output.grad.detach().float().square().mean(dim=-1)
                if use_teacher_forcing:
                    if prompt_mask is None or prompt_len is None:
                        raise RuntimeError("Internal error: teacher-forcing prompt metadata is missing.")
                    sensitivity = _collapse_teacher_forcing_prompt_sensitivity(
                        sensitivity, prompt_mask=prompt_mask, prompt_len=prompt_len
                    )
                weights = normalize_gradient_token_weights(
                    sensitivity, attention_mask=attention_mask, config=config
                )
                raw_weights[layer_idx][module_name].append(weights.detach().cpu())
            model.zero_grad(set_to_none=True)
    finally:
        for handle in handles:
            handle.remove()
        for parameter, requires_grad in zip(model.parameters(), parameter_requires_grad):
            parameter.requires_grad_(requires_grad)
        model.train(was_training)

    for layer_idx, weights_by_linear in raw_weights.items():
        for module_name, weights in weights_by_linear.items():
            if len(weights) != len(model_batches):
                raise RuntimeError(
                    f"Missing output gradients for layer {layer_idx}, Linear {module_name!r}: "
                    f"got {len(weights)} batches, expected {len(model_batches)}."
                )
    return group_token_weight_batches_by_linear(
        token_weights_by_layer_and_linear=raw_weights,
        token_group_batches=token_group_batches,
        model_batches=model_batches,
    )


def group_token_weight_batches_by_linear(
    *,
    token_weights_by_layer_and_linear: Mapping[int, Mapping[str, Sequence[torch.Tensor]]],
    token_group_batches: Sequence[torch.Tensor],
    model_batches: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, list[torch.Tensor]]]:
    """Replace each Linear token weights by its own global group means."""
    if len(token_group_batches) != len(model_batches):
        raise ValueError("token_group_batches length does not match model_batches length.")
    result: dict[int, dict[str, list[torch.Tensor]]] = {}
    for layer_idx, weights_by_linear in token_weights_by_layer_and_linear.items():
        result[int(layer_idx)] = {}
        for module_name, token_weight_batches in weights_by_linear.items():
            result[int(layer_idx)][module_name] = _group_weight_batches(
                token_weight_batches=token_weight_batches,
                token_group_batches=token_group_batches,
                model_batches=model_batches,
                scope=f"layer {layer_idx}, Linear {module_name}",
            )
    return result


def _group_weight_batches(
    *,
    token_weight_batches: Sequence[torch.Tensor],
    token_group_batches: Sequence[torch.Tensor],
    model_batches: Sequence[Mapping[str, Any]],
    scope: str,
) -> list[torch.Tensor]:
    if len(token_weight_batches) != len(token_group_batches):
        raise ValueError(f"{scope} weight batches do not match token group batches.")
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    prepared: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for weights, groups, batch in zip(token_weight_batches, token_group_batches, model_batches):
        weight_cpu = weights.detach().float().cpu()
        group_cpu = groups.detach().long().cpu()
        if weight_cpu.shape != group_cpu.shape:
            raise ValueError(f"{scope} token weights and groups have different shapes.")
        mask = _valid_token_mask(batch, shape=weight_cpu.shape)
        prepared.append((weight_cpu, group_cpu, mask))
        for group_id in torch.unique(group_cpu[mask]).tolist():
            group_mask = mask & (group_cpu == int(group_id))
            sums[int(group_id)] = sums.get(int(group_id), 0.0) + float(weight_cpu[group_mask].sum().item())
            counts[int(group_id)] = counts.get(int(group_id), 0) + int(group_mask.sum().item())

    means = {group_id: sums[group_id] / float(counts[group_id]) for group_id in sums if counts[group_id] > 0}
    grouped_batches: list[torch.Tensor] = []
    for weights, groups, mask in prepared:
        grouped = torch.zeros_like(weights)
        for group_id, group_mean in means.items():
            grouped = torch.where(groups == int(group_id), torch.full_like(grouped, float(group_mean)), grouped)
        grouped_batches.append(torch.where(mask, grouped, torch.zeros_like(grouped)))
    return grouped_batches


def _valid_token_mask(batch: Mapping[str, Any], *, shape: torch.Size) -> torch.Tensor:
    attention_mask = batch.get("attention_mask") if isinstance(batch, Mapping) else None
    if torch.is_tensor(attention_mask):
        mask = attention_mask.detach().bool().cpu()
        if mask.shape != shape:
            raise ValueError(f"attention_mask shape {tuple(mask.shape)} does not match token weight shape {tuple(shape)}")
        return mask
    return torch.ones(shape, dtype=torch.bool)
