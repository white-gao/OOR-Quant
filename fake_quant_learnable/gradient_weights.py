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
    target_token_ids: Sequence[torch.Tensor | int] | None = None,
    teacher_forcing_target_token_ids: Sequence[Sequence[torch.Tensor | int]] | None = None,
    config: GradientTokenWeightConfig = DEFAULT_GRADIENT_TOKEN_WEIGHT_CONFIG,
) -> dict[int, list[torch.Tensor]]:
    """Collect per-layer token weights from CE-loss gradient sensitivity.

    The default loss is the original first-SID-token CE at the last prompt
    position. When ``teacher_forcing_target_token_ids`` is provided, the loss is
    the average CE over multiple full-SID target sequences appended to the same
    prompt with teacher forcing; returned weights are still cropped back to the
    original prompt tokens so GPTQ Hessian collection stays prompt-prefill only.
    """
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
        raise ValueError(
            f"gradient target length {len(expected_targets)} does not match batches length {len(model_batches)}"
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
        if use_teacher_forcing:
            assert teacher_forcing_target_token_ids is not None
            iterator = zip(model_batches, teacher_forcing_target_token_ids)
        else:
            assert target_token_ids is not None
            iterator = zip(model_batches, target_token_ids)

        for batch, target_info in iterator:
            captured.clear()
            model.zero_grad(set_to_none=True)
            if use_teacher_forcing:
                expanded_batch, prompt_mask, prompt_len, loss_specs = _build_teacher_forcing_batch(
                    batch,
                    target_info,  # type: ignore[arg-type]
                )
                outputs = _forward_model(model, expanded_batch)
                logits = _extract_logits(outputs)
                loss = _teacher_forcing_full_sid_loss(logits, loss_specs)
                attention_mask = batch.get("attention_mask") if isinstance(batch, Mapping) else None
            else:
                outputs = _forward_model(model, batch)
                logits = _extract_logits(outputs)
                target = _target_tensor(target_info, logits=logits)  # type: ignore[arg-type]
                positions = _last_prompt_positions(batch, logits=logits)
                batch_indices = torch.arange(logits.shape[0], device=logits.device)
                loss = F.cross_entropy(logits[batch_indices, positions, :].float(), target)
                attention_mask = batch.get("attention_mask") if isinstance(batch, Mapping) else None
                prompt_mask = None
                prompt_len = None
            loss.backward()

            for layer_idx in selected:
                hidden = captured.get(layer_idx)
                if hidden is None or hidden.grad is None:
                    raise RuntimeError(f"No gradient was captured for layer {layer_idx}.")
                sensitivity = (hidden.detach().float() * hidden.grad.detach().float()).abs().mean(dim=-1)
                if use_teacher_forcing:
                    if prompt_mask is None or prompt_len is None:
                        raise RuntimeError("Internal error: teacher-forcing prompt metadata is missing.")
                    sensitivity = _collapse_teacher_forcing_prompt_sensitivity(
                        sensitivity,
                        prompt_mask=prompt_mask,
                        prompt_len=prompt_len,
                    )
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
    target_token_ids: Sequence[torch.Tensor | int] | None = None,
    token_group_batches: Sequence[torch.Tensor],
    teacher_forcing_target_token_ids: Sequence[Sequence[torch.Tensor | int]] | None = None,
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
        teacher_forcing_target_token_ids=teacher_forcing_target_token_ids,
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


def _build_teacher_forcing_batch(
    batch: Mapping[str, Any],
    target_sequences: Sequence[torch.Tensor | int],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int, list[tuple[int, torch.Tensor, torch.Tensor]]]:
    input_ids = batch.get("input_ids") if isinstance(batch, Mapping) else None
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise ValueError("Teacher-forcing gradient targets require input_ids with shape [batch, seq].")
    if input_ids.shape[0] != 1:
        raise ValueError("Teacher-forcing gradient target collection currently expects one prompt per batch.")

    prompt_mask = _valid_token_mask(batch, shape=input_ids.shape).to(device=input_ids.device)
    prompt_ids = input_ids[0][prompt_mask[0]].detach()
    prompt_len = int(prompt_ids.numel())
    if prompt_len == 0:
        raise ValueError("Cannot build teacher-forcing batch from an empty prompt.")

    prepared_targets: list[torch.Tensor] = []
    for target in target_sequences:
        target_ids = torch.as_tensor(target, device=input_ids.device, dtype=input_ids.dtype).reshape(-1)
        if target_ids.numel() == 0:
            continue
        prepared_targets.append(target_ids.detach())
    if not prepared_targets:
        raise ValueError("At least one non-empty teacher-forcing target sequence is required.")

    max_len = max(prompt_len + int(target.numel()) for target in prepared_targets)
    expanded_input_ids = torch.zeros(
        (len(prepared_targets), max_len),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    expanded_attention_mask = torch.zeros(
        (len(prepared_targets), max_len),
        dtype=torch.long,
        device=input_ids.device,
    )
    loss_specs: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for row_idx, target_ids in enumerate(prepared_targets):
        seq = torch.cat([prompt_ids, target_ids], dim=0)
        expanded_input_ids[row_idx, : seq.numel()] = seq
        expanded_attention_mask[row_idx, : seq.numel()] = 1
        positions = prompt_len - 1 + torch.arange(target_ids.numel(), device=input_ids.device)
        loss_specs.append((row_idx, positions, target_ids.to(dtype=torch.long)))

    return {
        "input_ids": expanded_input_ids,
        "attention_mask": expanded_attention_mask,
    }, prompt_mask, prompt_len, loss_specs


def _teacher_forcing_full_sid_loss(
    logits: torch.Tensor,
    loss_specs: Sequence[tuple[int, torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    losses = []
    for row_idx, positions, target_ids in loss_specs:
        row_logits = logits[row_idx, positions.to(device=logits.device), :].float()
        losses.append(F.cross_entropy(row_logits, target_ids.to(device=logits.device), reduction="mean"))
    if not losses:
        raise ValueError("No teacher-forcing loss terms were built.")
    return torch.stack(losses).mean()


def _collapse_teacher_forcing_prompt_sensitivity(
    sensitivity: torch.Tensor,
    *,
    prompt_mask: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    if sensitivity.ndim != 2:
        raise ValueError(f"Expected teacher-forcing sensitivity shape [targets, seq], got {tuple(sensitivity.shape)}")
    prompt_sensitivity = sensitivity[:, :prompt_len].mean(dim=0)
    collapsed = torch.zeros(prompt_mask.shape, device=sensitivity.device, dtype=sensitivity.dtype)
    collapsed[prompt_mask.to(device=sensitivity.device)] = prompt_sensitivity.reshape(-1)
    return collapsed


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
