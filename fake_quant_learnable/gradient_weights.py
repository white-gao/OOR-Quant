from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GradientTokenWeightConfig:
    clip_percentile: float = 99.0
    weight_floor: float = 0.05
    normalize_mean: bool = True
    eps: float = 1e-12

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_GRADIENT_TOKEN_WEIGHT_CONFIG = GradientTokenWeightConfig()


def collect_gradient_token_weight_batches_by_layer(
    *,
    model: nn.Module,
    layers: Sequence[nn.Module],
    layer_indices: Sequence[int],
    model_batches: Sequence[Mapping[str, Any]],
    target_token_ids: Sequence[torch.Tensor | int],
    config: GradientTokenWeightConfig = DEFAULT_GRADIENT_TOKEN_WEIGHT_CONFIG,
) -> dict[int, list[torch.Tensor]]:
    """Collect per-layer token weights from CE-loss gradient sensitivity.

    The sensitivity for token t is mean_c(|hidden[t, c] * grad[t, c]|).
    These weights are calibration-only inputs for GPTQ Hessian accumulation.
    """
    selected = sorted(set(int(idx) for idx in layer_indices))
    if not selected:
        return {}
    if len(model_batches) != len(target_token_ids):
        raise ValueError(
            f"target_token_ids length {len(target_token_ids)} does not match batches length {len(model_batches)}"
        )
    for idx in selected:
        if idx < 0 or idx >= len(layers):
            raise ValueError(f"Layer index {idx} out of range for {len(layers)} layers.")

    was_training = model.training
    param_requires_grad = [param.requires_grad for param in model.parameters()]
    for param in model.parameters():
        param.requires_grad_(False)

    results: dict[int, list[torch.Tensor]] = {idx: [] for idx in selected}
    min_selected = selected[0]
    selected_set = set(selected)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_pre_hook(layer_idx: int):
        def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
            if args:
                hidden = args[0]
                if not torch.is_tensor(hidden):
                    return None
                if layer_idx == min_selected:
                    hidden = hidden.detach().requires_grad_(True)
                    new_args = (hidden, *args[1:])
                    captured[layer_idx] = hidden
                    return new_args, kwargs
                if layer_idx in selected_set:
                    hidden.retain_grad()
                    captured[layer_idx] = hidden
                return None

            key = "hidden_states" if "hidden_states" in kwargs else "input"
            hidden = kwargs.get(key)
            if not torch.is_tensor(hidden):
                return None
            if layer_idx == min_selected:
                hidden = hidden.detach().requires_grad_(True)
                new_kwargs = dict(kwargs)
                new_kwargs[key] = hidden
                captured[layer_idx] = hidden
                return args, new_kwargs
            if layer_idx in selected_set:
                hidden.retain_grad()
                captured[layer_idx] = hidden
            return None

        return hook

    for layer_idx in selected:
        handles.append(layers[layer_idx].register_forward_pre_hook(make_pre_hook(layer_idx), with_kwargs=True))

    model.eval()
    try:
        for batch, target_token_id in zip(model_batches, target_token_ids):
            captured.clear()
            model.zero_grad(set_to_none=True)
            outputs = _forward_model(model, batch)
            logits = _extract_logits(outputs)
            target = _target_tensor(target_token_id, logits=logits)
            positions = _last_prompt_positions(batch, logits=logits)
            batch_indices = torch.arange(logits.shape[0], device=logits.device)
            loss = F.cross_entropy(logits[batch_indices, positions, :].float(), target)
            loss.backward()

            attention_mask = batch.get("attention_mask") if isinstance(batch, Mapping) else None
            for layer_idx in selected:
                hidden = captured.get(layer_idx)
                if hidden is None or hidden.grad is None:
                    raise RuntimeError(f"No gradient was captured for layer {layer_idx}.")
                sensitivity = (hidden.detach().float() * hidden.grad.detach().float()).abs().mean(dim=-1)
                weights = normalize_gradient_token_weights(
                    sensitivity,
                    attention_mask=attention_mask,
                    config=config,
                )
                results[layer_idx].append(weights.detach().cpu())
            model.zero_grad(set_to_none=True)
    finally:
        for handle in handles:
            handle.remove()
        for param, requires_grad in zip(model.parameters(), param_requires_grad):
            param.requires_grad_(requires_grad)
        model.train(was_training)

    return results


def collect_gradient_group_token_weight_batches_by_layer(
    *,
    model: nn.Module,
    layers: Sequence[nn.Module],
    layer_indices: Sequence[int],
    model_batches: Sequence[Mapping[str, Any]],
    target_token_ids: Sequence[torch.Tensor | int],
    token_group_batches: Sequence[torch.Tensor],
    config: GradientTokenWeightConfig = DEFAULT_GRADIENT_TOKEN_WEIGHT_CONFIG,
) -> dict[int, list[torch.Tensor]]:
    """Collect layer-wise group-smoothed gradient token weights.

    This keeps the same gradient signal as token-granular weighting, but replaces
    every token weight with its layer/global prompt-role group mean over the
    calibration set. The returned tensors are still per-token so GPTQ Hessian
    collection can use the existing token_weight_batches_by_layer interface.
    """
    if len(token_group_batches) != len(model_batches):
        raise ValueError(
            f"token_group_batches length {len(token_group_batches)} does not match batches length {len(model_batches)}"
        )

    token_weights_by_layer = collect_gradient_token_weight_batches_by_layer(
        model=model,
        layers=layers,
        layer_indices=layer_indices,
        model_batches=model_batches,
        target_token_ids=target_token_ids,
        config=config,
    )
    return group_token_weight_batches_by_layer(
        token_weights_by_layer=token_weights_by_layer,
        token_group_batches=token_group_batches,
        model_batches=model_batches,
    )


def group_token_weight_batches_by_layer(
    *,
    token_weights_by_layer: Mapping[int, Sequence[torch.Tensor]],
    token_group_batches: Sequence[torch.Tensor],
    model_batches: Sequence[Mapping[str, Any]],
) -> dict[int, list[torch.Tensor]]:
    if len(token_group_batches) != len(model_batches):
        raise ValueError(
            f"token_group_batches length {len(token_group_batches)} does not match batches length {len(model_batches)}"
        )

    grouped_by_layer: dict[int, list[torch.Tensor]] = {}
    for layer_idx, token_weight_batches in token_weights_by_layer.items():
        if len(token_weight_batches) != len(token_group_batches):
            raise ValueError(
                f"Layer {layer_idx} has {len(token_weight_batches)} token weight batches, "
                f"expected {len(token_group_batches)}"
            )

        group_sums: dict[int, float] = {}
        group_counts: dict[int, int] = {}
        prepared: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for weights, groups, batch in zip(token_weight_batches, token_group_batches, model_batches):
            weight_cpu = weights.detach().float().cpu()
            group_cpu = groups.detach().long().cpu()
            if group_cpu.shape != weight_cpu.shape:
                raise ValueError(
                    f"Token group shape {tuple(group_cpu.shape)} does not match "
                    f"token weights {tuple(weight_cpu.shape)} for layer {layer_idx}."
                )
            mask = _valid_token_mask(batch, shape=weight_cpu.shape).cpu()
            prepared.append((weight_cpu, group_cpu, mask))
            for group_id in torch.unique(group_cpu[mask]).tolist():
                group_mask = mask & (group_cpu == int(group_id))
                group_sums[int(group_id)] = group_sums.get(int(group_id), 0.0) + float(weight_cpu[group_mask].sum().item())
                group_counts[int(group_id)] = group_counts.get(int(group_id), 0) + int(group_mask.sum().item())

        group_means = {
            group_id: group_sums[group_id] / float(group_counts[group_id])
            for group_id in group_sums
            if group_counts[group_id] > 0
        }

        grouped_batches: list[torch.Tensor] = []
        for weights, groups, mask in prepared:
            grouped = torch.zeros_like(weights)
            for group_id, mean in group_means.items():
                grouped = torch.where(groups == int(group_id), torch.full_like(grouped, float(mean)), grouped)
            grouped = torch.where(mask, grouped, torch.zeros_like(grouped))
            grouped_batches.append(grouped)
        grouped_by_layer[int(layer_idx)] = grouped_batches
    return grouped_by_layer


def _valid_token_mask(batch: Mapping[str, Any], *, shape: torch.Size) -> torch.Tensor:
    attention_mask = batch.get("attention_mask") if isinstance(batch, Mapping) else None
    if torch.is_tensor(attention_mask):
        mask = attention_mask.detach().bool().cpu()
        if mask.shape != shape:
            raise ValueError(f"attention_mask shape {tuple(mask.shape)} does not match token weight shape {tuple(shape)}")
        return mask
    return torch.ones(shape, dtype=torch.bool)



def normalize_gradient_token_weights(
    sensitivity: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    config: GradientTokenWeightConfig = DEFAULT_GRADIENT_TOKEN_WEIGHT_CONFIG,
) -> torch.Tensor:
    if sensitivity.ndim != 2:
        raise ValueError(f"Expected sensitivity shape [batch, seq], got {tuple(sensitivity.shape)}")
    weights = sensitivity.detach().float().clamp_min(float(config.eps))
    if attention_mask is None:
        mask = torch.ones_like(weights, dtype=torch.bool)
    else:
        mask = attention_mask.to(device=weights.device).bool()
        if mask.shape != weights.shape:
            raise ValueError(f"attention_mask shape {tuple(mask.shape)} does not match weights {tuple(weights.shape)}")

    valid = weights[mask]
    if valid.numel() == 0:
        return torch.zeros_like(weights)

    if config.clip_percentile < 100.0:
        q = max(0.0, min(float(config.clip_percentile), 100.0)) / 100.0
        clip_value = torch.quantile(valid, q).clamp_min(float(config.eps))
        weights = torch.clamp(weights, max=clip_value)

    weights = torch.where(mask, weights, torch.zeros_like(weights))
    if config.normalize_mean:
        weights = _normalize_valid_mean(weights, mask, eps=config.eps)
    if config.weight_floor > 0.0:
        floored = torch.clamp(weights, min=float(config.weight_floor))
        weights = torch.where(mask, floored, torch.zeros_like(weights))
        if config.normalize_mean:
            weights = _normalize_valid_mean(weights, mask, eps=config.eps)
    return weights


def _normalize_valid_mean(weights: torch.Tensor, mask: torch.Tensor, *, eps: float) -> torch.Tensor:
    valid = weights[mask]
    if valid.numel() == 0:
        return weights
    mean = valid.float().mean().clamp_min(float(eps))
    normalized = weights / mean
    return torch.where(mask, normalized, torch.zeros_like(normalized))


def _forward_model(model: nn.Module, batch: Mapping[str, Any]) -> Any:
    try:
        return model(**batch, use_cache=False)
    except TypeError:
        return model(**batch)


def _extract_logits(outputs: Any) -> torch.Tensor:
    logits = getattr(outputs, "logits", None)
    if torch.is_tensor(logits):
        return logits
    if isinstance(outputs, (tuple, list)) and outputs and torch.is_tensor(outputs[0]):
        return outputs[0]
    raise TypeError(f"Could not extract logits from output type {type(outputs)!r}.")


def _last_prompt_positions(batch: Mapping[str, Any], *, logits: torch.Tensor) -> torch.Tensor:
    attention_mask = batch.get("attention_mask") if isinstance(batch, Mapping) else None
    if torch.is_tensor(attention_mask):
        return attention_mask.to(device=logits.device).long().sum(dim=-1).clamp_min(1) - 1
    return torch.full((logits.shape[0],), logits.shape[1] - 1, device=logits.device, dtype=torch.long)


def _target_tensor(target_token_id: torch.Tensor | int, *, logits: torch.Tensor) -> torch.Tensor:
    target = torch.as_tensor(target_token_id, device=logits.device, dtype=torch.long).reshape(-1)
    if target.numel() == 1 and logits.shape[0] != 1:
        target = target.expand(logits.shape[0])
    if target.numel() != logits.shape[0]:
        raise ValueError(f"Expected {logits.shape[0]} target ids, got {target.numel()}.")
    return target
