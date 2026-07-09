from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from fake_quant_learnable.gptq import collect_gptq_hessians
from fake_quant_learnable.support.runtime_utils import _detach_tree, _module_device, _move_tree_to_device
from fake_quant_learnable.support.smoothquant_runtime import Batch, _batch_to_args_kwargs

from .apply import NaiveW8A8Summary, apply_naive_w8a8
from .modules import ActivationQuantMode


WEIGHT_QUANT_MODES = (
    "minmax",
    "gptq",
    "weighted_gptq",
    "grad_weighted_gptq",
    "slot_weighted_gptq",
    "slot_grad_weighted_gptq",
)

SID_ITEM_RE = re.compile(
    r"<\|sid_begin\|>"
    r"(?P<sid><s_a_[^>]+><s_b_[^>]+><s_c_[^>]+>)"
    r"<\|sid_end\|>"
)


def default_calib_split(
    data_dir: str | Path,
    fallback_split: str,
    *,
    task_name: str,
    resolve_path,
) -> str:
    calib_file = resolve_path(data_dir) / task_name / f"{task_name}_calib.parquet"
    return "calib" if calib_file.exists() else fallback_split


def format_prompt(prompt: str, prompt_token: str) -> str:
    if prompt_token and not prompt.endswith(prompt_token):
        return prompt + prompt_token
    return prompt


def extract_sid_teacher_forcing_targets(ground_truth: str, *, max_items: int = 1) -> list[str]:
    if max_items <= 0:
        return []
    targets: list[str] = []
    for match in SID_ITEM_RE.finditer(ground_truth or ""):
        targets.append(match.group("sid"))
        if len(targets) >= max_items:
            break
    return targets


def build_first_sid_target_token_ids(tokenizer: Any, samples: Sequence[Mapping[str, Any]]) -> list[torch.Tensor]:
    target_ids: list[torch.Tensor] = []
    for sample_idx, sample in enumerate(samples):
        targets = extract_sid_teacher_forcing_targets(str(sample.get("ground_truth", "")), max_items=1)
        if not targets:
            raise ValueError(f"Calibration sample {sample_idx} does not contain a ground-truth SID target.")
        encoded = tokenizer(targets[0], add_special_tokens=False, return_tensors="pt")
        ids = encoded["input_ids"].reshape(-1)
        if ids.numel() == 0:
            raise ValueError(f"Calibration sample {sample_idx} produced an empty SID target tokenization.")
        target_ids.append(ids[0].detach().cpu())
    return target_ids


def build_sid_teacher_forcing_target_token_ids(
    tokenizer: Any,
    samples: Sequence[Mapping[str, Any]],
    *,
    max_items: int,
) -> list[list[torch.Tensor]]:
    if max_items <= 0:
        raise ValueError(f"max_items must be positive, got {max_items}.")
    all_target_ids: list[list[torch.Tensor]] = []
    for sample_idx, sample in enumerate(samples):
        targets = extract_sid_teacher_forcing_targets(str(sample.get("ground_truth", "")), max_items=max_items)
        if not targets:
            raise ValueError(f"Calibration sample {sample_idx} does not contain a ground-truth SID target.")
        sample_target_ids: list[torch.Tensor] = []
        for target in targets:
            encoded = tokenizer(target, add_special_tokens=False, return_tensors="pt")
            ids = encoded["input_ids"].reshape(-1)
            if ids.numel() == 0:
                raise ValueError(f"Calibration sample {sample_idx} produced an empty SID target tokenization.")
            sample_target_ids.append(ids.detach().cpu())
        all_target_ids.append(sample_target_ids)
    return all_target_ids


def build_model_batches(
    *,
    tokenizer: Any,
    prompts: Sequence[str],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        batches.append(
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in encoded.items()
            }
        )
    return batches


def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise AttributeError("Could not find transformer layers at model.model.layers or model.layers.")
    if not isinstance(layers, nn.ModuleList):
        raise TypeError(f"Expected layers to be nn.ModuleList, got {type(layers)!r}.")
    return layers


def parse_layer_indices(spec: str, *, num_layers: int) -> list[int]:
    spec = spec.strip()
    if spec == "all":
        return list(range(num_layers))
    if spec.startswith("last:"):
        count = int(spec.split(":", 1)[1])
        if count <= 0 or count > num_layers:
            raise ValueError(f"Invalid last layer count: {count}")
        return list(range(num_layers - count, num_layers))

    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid layer range: {part}")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))

    deduped = sorted(set(indices))
    if not deduped:
        raise ValueError("No layer indices selected.")
    for idx in deduped:
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {idx} out of range for {num_layers} layers.")
    return deduped


def _first_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"Could not find a tensor in output type {type(value)!r}.")


def capture_layer_input_batches(
    *,
    model: nn.Module,
    layer: nn.Module,
    model_batches: Iterable[Mapping[str, Any]],
) -> list[Batch]:
    captured: list[Batch] = []

    def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        captured.append((_detach_tree(args), _detach_tree(kwargs)))

    handle = layer.register_forward_pre_hook(hook, with_kwargs=True)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in model_batches:
                try:
                    model(**batch, use_cache=False)
                except TypeError:
                    model(**batch)
    finally:
        handle.remove()
        model.train(was_training)
    return captured


def advance_layer_input_batches(
    *,
    layer: nn.Module,
    batches: Sequence[Batch],
) -> list[Batch]:
    advanced: list[Batch] = []
    was_training = layer.training
    target_device = _module_device(layer)
    layer.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                args, kwargs = _batch_to_args_kwargs(batch)
                args = _move_tree_to_device(args, target_device)
                kwargs = _move_tree_to_device(kwargs, target_device)
                output = layer(*args, **kwargs)
                hidden = _first_tensor(output).detach()
                next_args: tuple[Any, ...] = (hidden, *args[1:]) if args else ()
                next_kwargs = dict(kwargs)
                if not next_args:
                    if "hidden_states" in next_kwargs:
                        next_kwargs["hidden_states"] = hidden
                    else:
                        next_args = (hidden,)
                advanced.append((_detach_tree(next_args), _detach_tree(next_kwargs)))
    finally:
        layer.train(was_training)
    return advanced


def apply_gptq_real_w8a8_layers(
    *,
    model: nn.Module,
    model_batches: Sequence[Mapping[str, Any]],
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
    token_weight_batches: Sequence[torch.Tensor] | None = None,
    token_weight_batches_by_layer: Mapping[int, Sequence[torch.Tensor]] | None = None,
) -> NaiveW8A8Summary:
    layers = get_transformer_layers(model)
    selected_layer_indices = sorted(layer_indices)
    fp_inputs: list[Batch] | None = None
    stream_layer_idx: int | None = None
    replaced = 0
    skipped = 0
    shared_attention = 0
    shared_mlp = 0

    for layer_idx in selected_layer_indices:
        if fp_inputs is None:
            fp_inputs = capture_layer_input_batches(
                model=model,
                layer=layers[layer_idx],
                model_batches=model_batches,
            )
            stream_layer_idx = layer_idx
        else:
            if stream_layer_idx is None:
                raise RuntimeError("Internal error: stream_layer_idx is not initialized.")
            while stream_layer_idx < layer_idx:
                fp_inputs = advance_layer_input_batches(
                    layer=layers[stream_layer_idx],
                    batches=fp_inputs,
                )
                stream_layer_idx += 1

        layer = layers[layer_idx]
        layer_token_weight_batches = (
            token_weight_batches_by_layer.get(layer_idx)
            if token_weight_batches_by_layer is not None
            else token_weight_batches
        )
        hessians = collect_gptq_hessians(
            layer,
            fp_inputs,
            token_weight_batches=layer_token_weight_batches,
        )
        next_fp_inputs = advance_layer_input_batches(layer=layer, batches=fp_inputs)
        summary = apply_naive_w8a8(
            layer,
            skip_module_names=(),
            target_regex=target_regex,
            skip_regex=skip_regex,
            output_dtype=output_dtype,
            use_fast_accum=use_fast_accum,
            activation_quant_mode=activation_quant_mode,
            decode_a16_when_single_token=decode_a16_when_single_token,
            activation_tail_tokens=activation_tail_tokens,
            gptq_hessians=hessians,
            gptq_damp_percent=damp_percent,
            gptq_block_size=block_size,
        )
        replaced += summary.replaced_linears
        skipped += summary.skipped_linears
        shared_attention += summary.shared_attention_modules
        shared_mlp += summary.shared_mlp_modules
        fp_inputs = next_fp_inputs
        stream_layer_idx = layer_idx + 1
        print(
            f"[hf_naive_w8a8] gptq layer={layer_idx} replaced_linears={summary.replaced_linears}, "
            f"damp_percent={damp_percent}, block_size={block_size}"
        )

    return NaiveW8A8Summary(
        replaced_linears=replaced,
        skipped_linears=skipped,
        shared_attention_modules=shared_attention,
        shared_mlp_modules=shared_mlp,
    )
