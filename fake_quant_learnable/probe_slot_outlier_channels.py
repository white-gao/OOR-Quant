from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from fake_quant_learnable.token_weights import (
    SLOT_TOKEN_GROUPS,
    build_prompt_slot_token_group_batches,
)
from real_quant.full_precision.generator import dtype_from_name
from real_quant.full_precision.run_hf_baseline import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_PATH,
    RECOMMENDATION_TASKS,
    load_task_data,
    parse_sample_size,
    resolve_repo_path,
)
from real_quant.naive_w8a8.gptq_runtime import (
    build_model_batches,
    default_calib_split,
    format_prompt,
    get_transformer_layers,
    parse_layer_indices,
)


DEFAULT_OUTPUT_ROOT = "fake_quant_learnable/results/analysis/slot_tokenwise_outlier_channels"
DEFAULT_LINEAR_REGEX = r"(q_proj|o_proj|gate_proj|down_proj)$"
MAIN_TOKEN_GROUPS = ("text", "sid_a", "sid_b", "sid_c")
SID_TOKEN_GROUPS = ("sid_a", "sid_b", "sid_c")
GROUP_COLORS = {
    "text": "#4D4D4D",
    "sid_a": "#D73027",
    "sid_b": "#2878B5",
    "sid_c": "#1A9850",
    "boundary": "#7B3294",
}
SELECTION_COLORS = {
    "text": "#E6E6E6",
    "sid_a": "#F94144",
    "sid_b": "#43AAE8",
    "sid_c": "#90BE6D",
    "boundary": "#C77DFF",
}


@dataclass
class RepresentativeActivation:
    activations: torch.Tensor
    groups: torch.Tensor


@dataclass
class TokenwiseOutlierAccumulator:
    """Stream per-token top-k activation-channel occurrence counts."""

    outlier_fraction: float
    channel_counts_by_split: dict[int, torch.Tensor] = field(default_factory=dict)
    token_count_by_split: dict[int, int] = field(default_factory=dict)
    topk: int | None = None
    num_channels: int | None = None

    def add(self, values: torch.Tensor, *, split_id: int) -> None:
        if split_id not in (0, 1):
            raise ValueError(f"split_id must be 0 or 1, got {split_id}.")
        prepared = values.detach().float().reshape(-1, values.shape[-1])
        if prepared.shape[0] == 0:
            return
        channels = int(prepared.shape[1])
        topk = _fraction_count(channels, self.outlier_fraction)
        if self.num_channels is not None and self.num_channels != channels:
            raise ValueError(f"Channel count changed from {self.num_channels} to {channels}.")
        self.num_channels = channels
        self.topk = topk
        indices = torch.topk(prepared.abs(), k=topk, dim=1, largest=True, sorted=False).indices
        current = torch.bincount(indices.reshape(-1), minlength=channels)
        previous = self.channel_counts_by_split.get(split_id)
        self.channel_counts_by_split[split_id] = current if previous is None else previous + current
        self.token_count_by_split[split_id] = self.token_count_by_split.get(split_id, 0) + int(
            prepared.shape[0]
        )

    def profile(self) -> dict[str, torch.Tensor | int]:
        if self.num_channels is None or self.topk is None or not self.channel_counts_by_split:
            raise ValueError("Cannot build a token-wise outlier profile with zero tokens.")
        template = next(iter(self.channel_counts_by_split.values()))
        split0 = self.channel_counts_by_split.get(0, torch.zeros_like(template))
        split1 = self.channel_counts_by_split.get(1, torch.zeros_like(template))
        counts = split0 + split1
        split0_tokens = int(self.token_count_by_split.get(0, 0))
        split1_tokens = int(self.token_count_by_split.get(1, 0))
        token_count = split0_tokens + split1_tokens
        return {
            "channel_counts": counts.detach().cpu(),
            "split0_channel_counts": split0.detach().cpu(),
            "split1_channel_counts": split1.detach().cpu(),
            "channel_frequency": (counts.float() / max(1, token_count)).detach().cpu(),
            "split0_channel_frequency": (split0.float() / max(1, split0_tokens)).detach().cpu(),
            "split1_channel_frequency": (split1.float() / max(1, split1_tokens)).detach().cpu(),
            "token_count": token_count,
            "split0_token_count": split0_tokens,
            "split1_token_count": split1_tokens,
            "topk": int(self.topk),
            "num_channels": int(self.num_channels),
        }


@dataclass
class TokenwiseOutlierProbeResult:
    outlier_fraction: float
    accumulators: dict[tuple[int, str, str], TokenwiseOutlierAccumulator] = field(default_factory=dict)
    representative: dict[tuple[int, str], RepresentativeActivation] = field(default_factory=dict)

    def add(
        self,
        *,
        layer: int,
        module: str,
        group: str,
        values: torch.Tensor,
        split_id: int,
    ) -> None:
        key = (int(layer), module, group)
        if key not in self.accumulators:
            self.accumulators[key] = TokenwiseOutlierAccumulator(self.outlier_fraction)
        self.accumulators[key].add(values, split_id=split_id)


def tokenwise_topk_mask(activations: torch.Tensor, *, fraction: float) -> torch.Tensor:
    if activations.ndim != 2:
        raise ValueError(f"Expected activations [tokens, channels], got {tuple(activations.shape)}.")
    topk = _fraction_count(int(activations.shape[1]), fraction)
    indices = torch.topk(activations.detach().float().abs(), k=topk, dim=1, sorted=False).indices
    mask = torch.zeros_like(activations, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return mask


def average_intra_token_overlap(
    channel_counts: torch.Tensor,
    *,
    token_count: int,
    topk: int,
) -> float:
    if token_count < 2 or topk <= 0:
        return 0.0
    counts = channel_counts.detach().double().reshape(-1)
    shared_channel_pairs = float((counts * (counts - 1.0) / 2.0).sum().item())
    token_pairs = token_count * (token_count - 1) / 2.0
    return _round_float(shared_channel_pairs / (token_pairs * topk))


def average_inter_token_overlap(
    counts_a: torch.Tensor,
    *,
    token_count_a: int,
    counts_b: torch.Tensor,
    token_count_b: int,
    topk: int,
) -> float:
    if token_count_a <= 0 or token_count_b <= 0 or topk <= 0:
        return 0.0
    a = counts_a.detach().double().reshape(-1)
    b = counts_b.detach().double().reshape(-1)
    if a.numel() != b.numel():
        raise ValueError("Channel-count vectors must have equal length.")
    shared = float(torch.dot(a, b).item())
    return _round_float(shared / (token_count_a * token_count_b * topk))


def collect_slot_outlier_channels(
    *,
    model: nn.Module,
    layers: Sequence[nn.Module],
    layer_indices: Sequence[int],
    plot_layer_indices: Sequence[int],
    model_batches: Sequence[Mapping[str, Any]],
    slot_group_batches: Sequence[torch.Tensor],
    split_ids: Sequence[int],
    representative_index: int,
    outlier_fraction: float,
    linear_regex: str = DEFAULT_LINEAR_REGEX,
    progress_every: int = 10,
) -> TokenwiseOutlierProbeResult:
    sample_count = len(model_batches)
    if len(slot_group_batches) != sample_count or len(split_ids) != sample_count:
        raise ValueError("model_batches, slot_group_batches, and split_ids must have equal lengths.")
    if not 0 <= representative_index < sample_count:
        raise ValueError(f"representative_index {representative_index} is outside [0, {sample_count}).")
    if any(int(split_id) not in (0, 1) for split_id in split_ids):
        raise ValueError("Every calibration split id must be 0 or 1.")

    selected = sorted(set(int(index) for index in layer_indices))
    plot_selected = set(int(index) for index in plot_layer_indices)
    pattern = re.compile(linear_regex)
    result = TokenwiseOutlierProbeResult(outlier_fraction=outlier_fraction)
    handles = []
    state: dict[str, Any] = {
        "groups": None,
        "valid_mask": None,
        "split_id": None,
        "sample_index": None,
        "captured": 0,
    }

    def make_hook(layer_idx: int, module_name: str, linear: nn.Linear):
        def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            x = args[0] if args else kwargs.get("input", kwargs.get("hidden_states"))
            groups = state["groups"]
            valid_mask = state["valid_mask"]
            split_id = state["split_id"]
            sample_index = state["sample_index"]
            if not torch.is_tensor(x) or not torch.is_tensor(groups) or not torch.is_tensor(valid_mask):
                return
            if split_id is None or sample_index is None:
                return
            if x.shape[-1] != linear.in_features:
                raise ValueError(
                    f"Expected input dim {linear.in_features} for layer={layer_idx}, module={module_name}; "
                    f"got {tuple(x.shape)}."
                )
            x2d = x.detach().float().reshape(-1, linear.in_features)
            group_flat = groups.to(device=x.device).reshape(-1)
            valid_flat = valid_mask.to(device=x.device).reshape(-1).bool()
            if x2d.shape[0] != group_flat.numel() or x2d.shape[0] != valid_flat.numel():
                raise ValueError(
                    f"Token masks have {group_flat.numel()} rows but activations have {x2d.shape[0]} "
                    f"for layer={layer_idx}, module={module_name}."
                )
            for group_id, group_name in enumerate(SLOT_TOKEN_GROUPS):
                mask = valid_flat & (group_flat == int(group_id))
                if mask.any():
                    result.add(
                        layer=layer_idx,
                        module=module_name,
                        group=group_name,
                        values=x2d[mask],
                        split_id=int(split_id),
                    )

            representative_key = (layer_idx, module_name)
            if (
                int(sample_index) == representative_index
                and layer_idx in plot_selected
                and representative_key not in result.representative
            ):
                result.representative[representative_key] = RepresentativeActivation(
                    activations=x2d[valid_flat].to(device="cpu", dtype=torch.float32),
                    groups=group_flat[valid_flat].to(device="cpu", dtype=torch.long),
                )
            state["captured"] = int(state["captured"]) + 1

        return hook

    for layer_idx in selected:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} is outside [0, {len(layers)}).")
        for module_name, module in layers[layer_idx].named_modules():
            if isinstance(module, nn.Linear) and pattern.search(module_name):
                handles.append(
                    module.register_forward_pre_hook(
                        make_hook(layer_idx, module_name, module),
                        with_kwargs=True,
                    )
                )
    if not handles:
        raise ValueError(f"No Linear modules matched regex {linear_regex!r} in layers {selected}.")

    was_training = model.training
    model.eval()
    forward_module = model.model if hasattr(model, "model") and hasattr(model.model, "layers") else model
    try:
        with torch.no_grad():
            for sample_index, (batch, groups, split_id) in enumerate(
                zip(model_batches, slot_group_batches, split_ids)
            ):
                state["groups"] = groups
                attention_mask = batch.get("attention_mask") if isinstance(batch, Mapping) else None
                state["valid_mask"] = (
                    attention_mask.bool()
                    if torch.is_tensor(attention_mask)
                    else torch.ones_like(groups, dtype=torch.bool)
                )
                state["split_id"] = int(split_id)
                state["sample_index"] = sample_index
                state["captured"] = 0
                try:
                    forward_module(**batch, use_cache=False)
                except TypeError:
                    forward_module(**batch)
                if int(state["captured"]) == 0:
                    raise RuntimeError("No Linear inputs were captured during model forward.")
                if progress_every > 0 and (
                    (sample_index + 1) % progress_every == 0 or sample_index + 1 == sample_count
                ):
                    print(
                        f"[probe_slot_outlier_channels] sample {sample_index + 1}/{sample_count} "
                        f"captured_modules={state['captured']}"
                    )
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return result


def build_probe_outputs(
    result: TokenwiseOutlierProbeResult,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, torch.Tensor],
]:
    profiles: dict[tuple[int, str, str], dict[str, torch.Tensor | int]] = {}
    group_rows: list[dict[str, Any]] = []
    profile_tensors: dict[str, torch.Tensor] = {}

    for key, accumulator in sorted(result.accumulators.items()):
        layer, module, group = key
        profile = accumulator.profile()
        profiles[key] = profile
        counts = _tensor_value(profile, "channel_counts")
        frequency = _tensor_value(profile, "channel_frequency")
        split0_frequency = _tensor_value(profile, "split0_channel_frequency")
        split1_frequency = _tensor_value(profile, "split1_channel_frequency")
        token_count = int(profile["token_count"])
        topk = int(profile["topk"])
        recurrent = top_fraction_indices(frequency, fraction=result.outlier_fraction)
        split0_recurrent = top_fraction_indices(split0_frequency, fraction=result.outlier_fraction)
        split1_recurrent = top_fraction_indices(split1_frequency, fraction=result.outlier_fraction)
        split_set_metrics = channel_set_metrics(split0_recurrent, split1_recurrent)
        concentration = float(counts[recurrent].sum().item()) / max(1, token_count * topk)

        profile_tensors[f"layer{layer}.{module}.channel_counts.{group}"] = counts
        profile_tensors[f"layer{layer}.{module}.channel_frequency.{group}"] = frequency
        profile_tensors[f"layer{layer}.{module}.split0_channel_frequency.{group}"] = split0_frequency
        profile_tensors[f"layer{layer}.{module}.split1_channel_frequency.{group}"] = split1_frequency
        group_rows.append(
            {
                "layer": layer,
                "module": module,
                "group": group,
                "token_count": token_count,
                "split0_token_count": int(profile["split0_token_count"]),
                "split1_token_count": int(profile["split1_token_count"]),
                "num_channels": int(profile["num_channels"]),
                "token_topk": topk,
                "outlier_fraction": result.outlier_fraction,
                "random_overlap_baseline": _round_float(topk / int(profile["num_channels"])),
                "intra_token_overlap": average_intra_token_overlap(
                    counts,
                    token_count=token_count,
                    topk=topk,
                ),
                "split_frequency_cosine": cosine_similarity(split0_frequency, split1_frequency),
                "split_recurrent_overlap": _round_float(float(split_set_metrics["overlap"])),
                "split_recurrent_jaccard": _round_float(float(split_set_metrics["jaccard"])),
                "recurrent_channel_concentration": _round_float(concentration),
            }
        )

    group_lookup = {
        (int(row["layer"]), str(row["module"]), str(row["group"])): row for row in group_rows
    }
    layer_modules = sorted({(layer, module) for layer, module, _group in profiles})
    pair_rows: list[dict[str, Any]] = []
    for layer, module in layer_modules:
        available = [group for group in SLOT_TOKEN_GROUPS if (layer, module, group) in profiles]
        for group_a, group_b in itertools.combinations(available, 2):
            profile_a = profiles[(layer, module, group_a)]
            profile_b = profiles[(layer, module, group_b)]
            counts_a = _tensor_value(profile_a, "channel_counts")
            counts_b = _tensor_value(profile_b, "channel_counts")
            frequency_a = _tensor_value(profile_a, "channel_frequency")
            frequency_b = _tensor_value(profile_b, "channel_frequency")
            topk = int(profile_a["topk"])
            inter = average_inter_token_overlap(
                counts_a,
                token_count_a=int(profile_a["token_count"]),
                counts_b=counts_b,
                token_count_b=int(profile_b["token_count"]),
                topk=topk,
            )
            intra_reference = _mean(
                [
                    float(group_lookup[(layer, module, group_a)]["intra_token_overlap"]),
                    float(group_lookup[(layer, module, group_b)]["intra_token_overlap"]),
                ]
            )
            recurrent_a = top_fraction_indices(frequency_a, fraction=result.outlier_fraction)
            recurrent_b = top_fraction_indices(frequency_b, fraction=result.outlier_fraction)
            recurrent_metrics = channel_set_metrics(recurrent_a, recurrent_b)
            pair_rows.append(
                {
                    "layer": layer,
                    "module": module,
                    "group_a": group_a,
                    "group_b": group_b,
                    "category": _pair_category(group_a, group_b),
                    "inter_token_overlap": inter,
                    "pair_intra_reference": intra_reference,
                    "pair_intra_minus_inter": _round_float(intra_reference - inter),
                    "frequency_cosine": cosine_similarity(frequency_a, frequency_b),
                    "recurrent_channel_overlap": _round_float(float(recurrent_metrics["overlap"])),
                    "recurrent_channel_jaccard": _round_float(float(recurrent_metrics["jaccard"])),
                }
            )

    layer_rows: list[dict[str, Any]] = []
    for layer, module in layer_modules:
        groups = [
            row for row in group_rows if int(row["layer"]) == layer and str(row["module"]) == module
        ]
        pairs = [
            row for row in pair_rows if int(row["layer"]) == layer and str(row["module"]) == module
        ]
        main_groups = [row for row in groups if str(row["group"]) in MAIN_TOKEN_GROUPS]
        text_pairs = [row for row in pairs if row["category"] == "text_vs_sid"]
        sid_pairs = [row for row in pairs if row["category"] == "sid_internal"]
        intra = _mean(float(row["intra_token_overlap"]) for row in main_groups)
        text_inter = _mean(float(row["inter_token_overlap"]) for row in text_pairs)
        sid_inter = _mean(float(row["inter_token_overlap"]) for row in sid_pairs)
        layer_rows.append(
            {
                "layer": layer,
                "module": module,
                "main_group_intra_token_overlap": intra,
                "text_vs_sid_inter_token_overlap": text_inter,
                "sid_internal_inter_token_overlap": sid_inter,
                "intra_minus_text_sid": _round_float(intra - text_inter),
                "intra_minus_sid_internal": _round_float(intra - sid_inter),
                "main_group_split_frequency_cosine": _mean(
                    float(row["split_frequency_cosine"]) for row in main_groups
                ),
                "main_group_recurrent_concentration": _mean(
                    float(row["recurrent_channel_concentration"]) for row in main_groups
                ),
            }
        )
    return group_rows, pair_rows, layer_rows, profile_tensors


def top_fraction_indices(profile: torch.Tensor, *, fraction: float) -> torch.Tensor:
    values = profile.detach().float().reshape(-1)
    if values.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    topk = _fraction_count(values.numel(), fraction)
    return torch.topk(values, k=topk, sorted=False).indices.sort().values.cpu()


def channel_set_metrics(first: torch.Tensor, second: torch.Tensor) -> dict[str, float | int]:
    first_set = {int(index) for index in first.detach().cpu().reshape(-1).tolist()}
    second_set = {int(index) for index in second.detach().cpu().reshape(-1).tolist()}
    intersection = len(first_set & second_set)
    union = len(first_set | second_set)
    denominator = min(len(first_set), len(second_set))
    return {
        "intersection": intersection,
        "overlap": intersection / float(denominator) if denominator else 0.0,
        "jaccard": intersection / float(union) if union else 0.0,
    }


def cosine_similarity(first: torch.Tensor, second: torch.Tensor) -> float:
    a = first.detach().double().reshape(-1)
    b = second.detach().double().reshape(-1)
    if a.numel() != b.numel():
        raise ValueError("Profile vectors must have equal length.")
    denominator = float(a.norm().item() * b.norm().item())
    return _round_float(float(torch.dot(a, b).item()) / denominator) if denominator else 0.0


def _pair_category(group_a: str, group_b: str) -> str:
    pair = {group_a, group_b}
    if "text" in pair and bool(pair & set(SID_TOKEN_GROUPS)):
        return "text_vs_sid"
    if group_a in SID_TOKEN_GROUPS and group_b in SID_TOKEN_GROUPS:
        return "sid_internal"
    if "boundary" in pair:
        return "with_boundary"
    return "other"


def _tensor_value(profile: Mapping[str, torch.Tensor | int], key: str) -> torch.Tensor:
    value = profile[key]
    if not torch.is_tensor(value):
        raise TypeError(f"Expected tensor profile {key!r}, got {type(value).__name__}.")
    return value


def _fraction_count(num_channels: int, fraction: float) -> int:
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}.")
    return max(1, min(int(num_channels), int(math.ceil(num_channels * fraction))))


def _mean(values: Sequence[float] | Any) -> float:
    materialized = [float(value) for value in values]
    return _round_float(sum(materialized) / len(materialized)) if materialized else 0.0


def _round_float(value: float) -> float:
    return float(round(float(value), 9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe token-wise slot activation outlier channels in OneRec."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--task", default="ad", choices=RECOMMENDATION_TASKS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--calib_split", default=None)
    parser.add_argument("--sample_size", default="128")
    parser.add_argument("--layers", default="all", help='Layer spec: "all", "last:K", or "0,2-4".')
    parser.add_argument(
        "--plot_layers",
        default="auto",
        help='Layers used for token-channel figures. "auto" selects early/middle/late layers.',
    )
    parser.add_argument("--linear_regex", default=DEFAULT_LINEAR_REGEX)
    parser.add_argument("--outlier_fraction", type=float, default=0.02)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--representative_index", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt_token", default="<|sid_begin|>")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.outlier_fraction <= 1.0:
        raise ValueError("--outlier_fraction must be in (0, 1].")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype_from_name(args.dtype),
    ).to(device)
    model.eval()
    calib_split = args.calib_split or default_calib_split(
        args.data_dir,
        args.split,
        task_name=args.task,
        resolve_path=resolve_repo_path,
    )
    calib_data = load_task_data(
        task_name=args.task,
        data_dir=str(resolve_repo_path(args.data_dir)),
        tokenizer=tokenizer,
        split=calib_split,
        sample_size=parse_sample_size(args.sample_size),
    )
    samples = list(calib_data.values())
    prompts = [format_prompt(sample["prompt"], args.prompt_token) for sample in samples]
    model_batches = build_model_batches(tokenizer=tokenizer, prompts=prompts, device=device)
    slot_group_batches = build_prompt_slot_token_group_batches(
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
    )
    layers = get_transformer_layers(model)
    layer_indices = parse_layer_indices(args.layers, num_layers=len(layers))
    plot_layers = _resolve_plot_layers(
        args.plot_layers,
        selected_layers=layer_indices,
        num_layers=len(layers),
    )
    split_ids = make_calibration_split_ids(len(samples), seed=args.split_seed)
    representative_index = resolve_representative_index(
        args.representative_index,
        model_batches=model_batches,
        slot_group_batches=slot_group_batches,
    )
    output_dir = _resolve_output_dir(
        args,
        sample_count=len(samples),
        outlier_fraction=args.outlier_fraction,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[probe_slot_outlier_channels] collecting token-wise profiles "
        f"task={args.task}, split={calib_split}, samples={len(samples)}, "
        f"layers={layer_indices}, token_top_fraction={args.outlier_fraction}, "
        f"representative_index={representative_index}"
    )
    result = collect_slot_outlier_channels(
        model=model,
        layers=layers,
        layer_indices=layer_indices,
        plot_layer_indices=plot_layers,
        model_batches=model_batches,
        slot_group_batches=slot_group_batches,
        split_ids=split_ids,
        representative_index=representative_index,
        outlier_fraction=args.outlier_fraction,
        linear_regex=args.linear_regex,
        progress_every=args.progress_every,
    )
    group_rows, pair_rows, layer_rows, profile_tensors = build_probe_outputs(result)
    _write_csv(output_dir / "group_tokenwise_summary.csv", group_rows)
    _write_csv(output_dir / "tokenwise_pair_overlap.csv", pair_rows)
    _write_csv(output_dir / "layer_tokenwise_summary.csv", layer_rows)
    torch.save(profile_tensors, output_dir / "tokenwise_channel_frequency.pt")
    plot_files = plot_probe_figures(
        result=result,
        profile_tensors=profile_tensors,
        pair_rows=pair_rows,
        layer_rows=layer_rows,
        output_dir=output_dir,
        plot_layers=plot_layers,
        representative_index=representative_index,
        outlier_fraction=args.outlier_fraction,
    )
    summary = {
        "task": args.task,
        "split": calib_split,
        "sample_size": len(samples),
        "model_path": args.model_path,
        "layers": layer_indices,
        "plot_layers": plot_layers,
        "linear_regex": args.linear_regex,
        "groups": list(SLOT_TOKEN_GROUPS),
        "main_groups": list(MAIN_TOKEN_GROUPS),
        "outlier_fraction": args.outlier_fraction,
        "split_seed": args.split_seed,
        "split_sample_counts": {str(value): split_ids.count(value) for value in (0, 1)},
        "representative_index": representative_index,
        "definition": "C_t = TopK_channel(abs(X[t,c]), K), computed independently for every token",
        "output_files": {
            "group_summary": "group_tokenwise_summary.csv",
            "pair_overlap": "tokenwise_pair_overlap.csv",
            "layer_summary": "layer_tokenwise_summary.csv",
            "profiles": "tokenwise_channel_frequency.pt",
            "report": "report.md",
            "plots": plot_files,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        build_report(
            summary=summary,
            group_rows=group_rows,
            pair_rows=pair_rows,
            layer_rows=layer_rows,
        ),
        encoding="utf-8",
    )
    print(f"[probe_slot_outlier_channels] saved to {output_dir}")


def make_calibration_split_ids(sample_count: int, *, seed: int) -> list[int]:
    indices = list(range(sample_count))
    random.Random(seed).shuffle(indices)
    split_ids = [1] * sample_count
    for index in indices[: sample_count // 2]:
        split_ids[index] = 0
    return split_ids


def resolve_representative_index(
    spec: str,
    *,
    model_batches: Sequence[Mapping[str, Any]],
    slot_group_batches: Sequence[torch.Tensor],
) -> int:
    if len(model_batches) != len(slot_group_batches) or not model_batches:
        raise ValueError("Representative selection requires equally sized non-empty batches and groups.")
    if spec != "auto":
        index = int(spec)
        if not 0 <= index < len(model_batches):
            raise ValueError(f"representative_index {index} is outside [0, {len(model_batches)}).")
        return index
    main_ids = {SLOT_TOKEN_GROUPS.index(group) for group in MAIN_TOKEN_GROUPS}
    candidates: list[tuple[int, int, int]] = []
    lengths: list[int] = []
    for index, (batch, groups) in enumerate(zip(model_batches, slot_group_batches)):
        attention_mask = batch.get("attention_mask")
        valid = attention_mask.bool() if torch.is_tensor(attention_mask) else torch.ones_like(groups).bool()
        length = int(valid.sum().item())
        coverage = len({int(value) for value in groups[valid].detach().cpu().tolist()} & main_ids)
        lengths.append(length)
        candidates.append((index, coverage, length))
    median_length = sorted(lengths)[len(lengths) // 2]
    max_coverage = max(item[1] for item in candidates)
    return min(
        [item for item in candidates if item[1] == max_coverage],
        key=lambda item: (abs(item[2] - median_length), item[0]),
    )[0]


def _resolve_plot_layers(spec: str, *, selected_layers: Sequence[int], num_layers: int) -> list[int]:
    if spec != "auto":
        requested = parse_layer_indices(spec, num_layers=num_layers)
        missing = [layer for layer in requested if layer not in set(selected_layers)]
        if missing:
            raise ValueError(f"Plot layers {missing} are not included in --layers.")
        return requested
    ordered = sorted(set(int(layer) for layer in selected_layers))
    return sorted({ordered[0], ordered[len(ordered) // 2], ordered[-1]}) if ordered else []


def _resolve_output_dir(
    args: argparse.Namespace,
    *,
    sample_count: int,
    outlier_fraction: float,
) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    model_tag = Path(str(args.model_path).rstrip("/")).name
    layer_tag = "all_layers" if args.layers == "all" else re.sub(r"[^A-Za-z0-9_-]+", "_", args.layers)
    fraction_tag = f"top{outlier_fraction * 100:g}pct"
    return Path(DEFAULT_OUTPUT_ROOT) / f"{args.task}_{model_tag}_s{sample_count}_{layer_tag}_{fraction_tag}"


def plot_probe_figures(
    *,
    result: TokenwiseOutlierProbeResult,
    profile_tensors: Mapping[str, torch.Tensor],
    pair_rows: Sequence[Mapping[str, Any]],
    layer_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    plot_layers: Sequence[int],
    representative_index: int,
    outlier_fraction: float,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[probe_slot_outlier_channels] matplotlib unavailable; skipping plots")
        return []
    files: list[str] = []
    files.extend(
        _plot_tokenwise_activation_maps(
            result=result,
            output_dir=output_dir,
            plot_layers=plot_layers,
            representative_index=representative_index,
            outlier_fraction=outlier_fraction,
            plt=plt,
        )
    )
    files.extend(
        _plot_group_frequency_maps(
            profile_tensors=profile_tensors,
            output_dir=output_dir,
            plot_layers=plot_layers,
            outlier_fraction=outlier_fraction,
            plt=plt,
        )
    )
    layerwise = _plot_layerwise_token_overlap(layer_rows=layer_rows, output_dir=output_dir, plt=plt)
    if layerwise:
        files.append(layerwise)
    scatter = _plot_token_overlap_scatter(pair_rows=pair_rows, output_dir=output_dir, plt=plt)
    if scatter:
        files.append(scatter)
    return files


def _plot_tokenwise_activation_maps(
    *,
    result: TokenwiseOutlierProbeResult,
    output_dir: Path,
    plot_layers: Sequence[int],
    representative_index: int,
    outlier_fraction: float,
    plt: Any,
) -> list[str]:
    import numpy as np
    from matplotlib.colors import ListedColormap, to_rgb
    from matplotlib.patches import Patch

    group_cmap = ListedColormap([GROUP_COLORS[group] for group in SLOT_TOKEN_GROUPS])
    files: list[str] = []
    for layer in plot_layers:
        modules = sorted(
            [module for candidate_layer, module in result.representative if candidate_layer == layer],
            key=_module_sort_key,
        )
        if not modules:
            continue
        fig = plt.figure(figsize=(16, max(3.2, 2.9 * len(modules))))
        grid = fig.add_gridspec(
            len(modules),
            3,
            width_ratios=(0.035, 1.12, 1.0),
            wspace=0.08,
            hspace=0.42,
        )
        for row_index, module in enumerate(modules):
            captured = result.representative[(layer, module)]
            activations = captured.activations.float()
            groups = captured.groups.long()
            outlier_mask = tokenwise_topk_mask(activations, fraction=outlier_fraction).cpu()
            log_magnitude = torch.log2(activations.abs() + 1e-6)
            sample = _sample_flat_values(log_magnitude, max_values=200_000)
            vmin = float(torch.quantile(sample, 0.02).item())
            vmax = float(torch.quantile(sample, 0.998).item())
            if vmax <= vmin:
                vmax = vmin + 1.0

            group_axis = fig.add_subplot(grid[row_index, 0])
            group_axis.imshow(
                groups.reshape(-1, 1).numpy(),
                aspect="auto",
                interpolation="nearest",
                cmap=group_cmap,
                vmin=0,
                vmax=len(SLOT_TOKEN_GROUPS) - 1,
            )
            group_axis.set_xticks([])
            group_axis.set_yticks([])

            magnitude_axis = fig.add_subplot(grid[row_index, 1])
            magnitude_axis.imshow(
                log_magnitude.numpy(),
                aspect="auto",
                interpolation="nearest",
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
            )
            magnitude_axis.set_title(f"{_module_display_name(module)}: log2 |activation|", fontsize=10)
            magnitude_axis.set_ylabel("token index")
            magnitude_axis.set_xlabel("input channel index")

            mask_axis = fig.add_subplot(grid[row_index, 2], sharey=magnitude_axis)
            rgb = np.full((outlier_mask.shape[0], outlier_mask.shape[1], 3), 0.035, dtype=np.float32)
            mask_numpy = outlier_mask.numpy()
            groups_numpy = groups.numpy()
            for group_id, group in enumerate(SLOT_TOKEN_GROUPS):
                rows = groups_numpy == group_id
                if rows.any():
                    selected_rgb = rgb[rows]
                    selected_rgb[mask_numpy[rows]] = np.asarray(
                        to_rgb(SELECTION_COLORS[group]),
                        dtype=np.float32,
                    )
                    rgb[rows] = selected_rgb
            mask_axis.imshow(rgb, aspect="auto", interpolation="nearest")
            mask_axis.set_title(f"per-token top-{outlier_fraction * 100:g}% |activation| channels", fontsize=10)
            mask_axis.set_xlabel("input channel index")
            mask_axis.tick_params(labelleft=False)

        legend = [Patch(facecolor=GROUP_COLORS[group], label=group) for group in SLOT_TOKEN_GROUPS]
        fig.legend(
            handles=legend,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=len(legend),
            frameon=False,
            fontsize=9,
        )
        fig.suptitle(
            f"Layer {layer}, sample {representative_index}: token-wise activation outlier locations",
            fontsize=12,
            y=0.997,
        )
        fig.subplots_adjust(top=0.90)
        path = output_dir / f"tokenwise_activation_outlier_layer_{layer:02d}.png"
        fig.savefig(path, dpi=190, bbox_inches="tight")
        plt.close(fig)
        files.append(path.name)
    return files


def _plot_group_frequency_maps(
    *,
    profile_tensors: Mapping[str, torch.Tensor],
    output_dir: Path,
    plot_layers: Sequence[int],
    outlier_fraction: float,
    plt: Any,
) -> list[str]:
    import numpy as np
    from matplotlib.colors import to_rgb

    files: list[str] = []
    unpacked = _unpack_profile_keys(profile_tensors)
    for layer in plot_layers:
        modules = sorted(
            {
                module
                for candidate_layer, module, kind, _group in unpacked
                if candidate_layer == layer and kind == "channel_frequency"
            },
            key=_module_sort_key,
        )
        if not modules:
            continue
        fig, axes = plt.subplots(2, len(modules), figsize=(5.0 * len(modules), 5.8), squeeze=False)
        image = None
        for column, module in enumerate(modules):
            keys = [f"layer{layer}.{module}.channel_frequency.{group}" for group in SLOT_TOKEN_GROUPS]
            if not all(key in profile_tensors for key in keys):
                axes[0, column].set_visible(False)
                axes[1, column].set_visible(False)
                continue
            matrix = torch.stack([profile_tensors[key].float() for key in keys])
            log_frequency = torch.log10(matrix.clamp_min(1e-5)).clamp(-5.0, 0.0)
            image = axes[0, column].imshow(
                log_frequency.numpy(),
                aspect="auto",
                interpolation="nearest",
                cmap="magma",
                vmin=-5.0,
                vmax=0.0,
            )
            axes[0, column].set_title(_module_display_name(module), fontsize=10)
            axes[0, column].set_yticks(range(len(SLOT_TOKEN_GROUPS)))
            axes[0, column].set_yticklabels(SLOT_TOKEN_GROUPS if column == 0 else [])

            rgb = np.full((len(SLOT_TOKEN_GROUPS), matrix.shape[1], 3), 0.035, dtype=np.float32)
            for group_id, group in enumerate(SLOT_TOKEN_GROUPS):
                indices = top_fraction_indices(matrix[group_id], fraction=outlier_fraction).numpy()
                rgb[group_id, indices] = np.asarray(to_rgb(SELECTION_COLORS[group]), dtype=np.float32)
            axes[1, column].imshow(rgb, aspect="auto", interpolation="nearest")
            axes[1, column].set_xlabel("input channel index")
            axes[1, column].set_yticks(range(len(SLOT_TOKEN_GROUPS)))
            axes[1, column].set_yticklabels(SLOT_TOKEN_GROUPS if column == 0 else [])
            axes[1, column].set_title(f"top-{outlier_fraction * 100:g}% recurrent channels", fontsize=9)

        fig.suptitle(
            f"Layer {layer}: frequency of entering per-token activation top-k",
            fontsize=12,
        )
        if image is not None:
            fig.colorbar(image, ax=list(axes[0]), shrink=0.8, label="log10(token occurrence frequency)")
        fig.subplots_adjust(top=0.86, hspace=0.62, wspace=0.22)
        path = output_dir / f"slot_channel_frequency_map_layer_{layer:02d}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        files.append(path.name)
    return files


def _plot_layerwise_token_overlap(
    *,
    layer_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    plt: Any,
) -> str | None:
    modules = sorted({str(row["module"]) for row in layer_rows}, key=_module_sort_key)
    if not modules:
        return None
    rows = int(math.ceil(len(modules) / 2.0))
    fig, axes = plt.subplots(rows, 2, figsize=(11, 3.6 * rows), squeeze=False)
    for axis, module in zip(axes.reshape(-1), modules):
        selected = sorted(
            [row for row in layer_rows if str(row["module"]) == module],
            key=lambda row: int(row["layer"]),
        )
        layers = [int(row["layer"]) for row in selected]
        axis.plot(
            layers,
            [float(row["main_group_intra_token_overlap"]) for row in selected],
            color="#111111",
            marker="o",
            markersize=3,
            linewidth=1.6,
            label="within group",
        )
        axis.plot(
            layers,
            [float(row["text_vs_sid_inter_token_overlap"]) for row in selected],
            color="#D73027",
            marker="o",
            markersize=3,
            linewidth=1.4,
            label="text vs SID",
        )
        axis.plot(
            layers,
            [float(row["sid_internal_inter_token_overlap"]) for row in selected],
            color="#2878B5",
            marker="o",
            markersize=3,
            linewidth=1.4,
            label="SID a/b/c",
        )
        axis.axhline(0.02, color="#777777", linestyle="--", linewidth=0.8, label="random 2%")
        axis.set_title(_module_display_name(module))
        axis.set_xlabel("layer")
        axis.set_ylabel("mean per-token top-k overlap")
        axis.set_ylim(-0.01, 1.01)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for axis in axes.reshape(-1)[len(modules) :]:
        axis.set_visible(False)
    fig.suptitle("Layer-wise within-group and cross-group token overlap", fontsize=12)
    path = output_dir / "layerwise_tokenwise_channel_overlap.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_token_overlap_scatter(
    *,
    pair_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    plt: Any,
) -> str | None:
    selected = [row for row in pair_rows if row["category"] in {"text_vs_sid", "sid_internal"}]
    if not selected:
        return None
    markers = {
        "self_attn.q_proj": "o",
        "self_attn.o_proj": "s",
        "mlp.gate_proj": "^",
        "mlp.down_proj": "D",
    }
    fig, axis = plt.subplots(figsize=(6.6, 6.0))
    for category, color, label in (
        ("text_vs_sid", "#D73027", "text vs SID"),
        ("sid_internal", "#2878B5", "SID a/b/c"),
    ):
        category_rows = [row for row in selected if row["category"] == category]
        for module in sorted({str(row["module"]) for row in category_rows}, key=_module_sort_key):
            rows = [row for row in category_rows if str(row["module"]) == module]
            axis.scatter(
                [float(row["inter_token_overlap"]) for row in rows],
                [float(row["pair_intra_reference"]) for row in rows],
                color=color,
                marker=markers.get(module, "o"),
                alpha=0.7,
                s=32,
                edgecolors="none",
                label=label if module == "self_attn.q_proj" else None,
            )
    axis.plot([0.0, 1.0], [0.0, 1.0], color="#666666", linestyle="--", linewidth=1.0)
    axis.set_xlim(-0.01, 1.01)
    axis.set_ylim(-0.01, 1.01)
    axis.set_xlabel("cross-group token overlap")
    axis.set_ylabel("mean within-group token overlap")
    axis.set_title("Within-group versus cross-group token-wise channel alignment")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    path = output_dir / "tokenwise_intra_vs_inter_overlap.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _sample_flat_values(values: torch.Tensor, *, max_values: int) -> torch.Tensor:
    flat = values.detach().float().reshape(-1).cpu()
    if flat.numel() <= max_values:
        return flat
    return flat[:: int(math.ceil(flat.numel() / max_values))]


def _unpack_profile_keys(
    profile_tensors: Mapping[str, torch.Tensor],
) -> list[tuple[int, str, str, str]]:
    pattern = re.compile(
        r"^layer(?P<layer>\d+)\.(?P<module>.+)\."
        r"(?P<kind>channel_counts|channel_frequency|split0_channel_frequency|split1_channel_frequency)\."
        r"(?P<group>[^.]+)$"
    )
    unpacked: list[tuple[int, str, str, str]] = []
    for key in profile_tensors:
        match = pattern.match(key)
        if match:
            unpacked.append(
                (
                    int(match.group("layer")),
                    match.group("module"),
                    match.group("kind"),
                    match.group("group"),
                )
            )
    return unpacked


def _module_sort_key(module: str) -> tuple[int, str]:
    order = {
        "self_attn.q_proj": 0,
        "self_attn.o_proj": 1,
        "mlp.gate_proj": 2,
        "mlp.down_proj": 3,
    }
    return order.get(module, len(order)), module


def _module_display_name(module: str) -> str:
    return {
        "self_attn.q_proj": "attention q/k/v input",
        "self_attn.o_proj": "attention output input",
        "mlp.gate_proj": "FFN gate/up input",
        "mlp.down_proj": "FFN down input",
    }.get(module, module)


def build_report(
    *,
    summary: Mapping[str, Any],
    group_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    layer_rows: Sequence[Mapping[str, Any]],
) -> str:
    main_groups = [row for row in group_rows if str(row["group"]) in MAIN_TOKEN_GROUPS]
    text_pairs = [row for row in pair_rows if row["category"] == "text_vs_sid"]
    sid_pairs = [row for row in pair_rows if row["category"] == "sid_internal"]
    mean_intra = _mean(float(row["intra_token_overlap"]) for row in main_groups)
    mean_text_inter = _mean(float(row["inter_token_overlap"]) for row in text_pairs)
    mean_sid_inter = _mean(float(row["inter_token_overlap"]) for row in sid_pairs)
    intra_over_text = sum(
        float(row["main_group_intra_token_overlap"])
        > float(row["text_vs_sid_inter_token_overlap"])
        for row in layer_rows
    )
    intra_over_sid = sum(
        float(row["main_group_intra_token_overlap"])
        > float(row["sid_internal_inter_token_overlap"])
        for row in layer_rows
    )
    lines = [
        "# Token-Wise Slot Activation Outlier Probe",
        "",
        f"- task: `{summary['task']}`",
        f"- split: `{summary['split']}`",
        f"- sample_size: `{summary['sample_size']}`",
        f"- calibration halves: `{summary['split_sample_counts']}`",
        f"- per-token top fraction: `{float(summary['outlier_fraction']) * 100:g}%`",
        f"- representative sample index: `{summary['representative_index']}`",
        f"- layers: `{summary['layers']}`",
        "",
        "## Definition",
        "",
        "```text",
        "C_t = TopK_channel(abs(X[t,c]), K)",
        "count[g,c] = sum_{t in group g} 1[c in C_t]",
        "frequency[g,c] = count[g,c] / N_g",
        "",
        "intra[g] = sum_c choose(count[g,c], 2) / (choose(N_g, 2) * K)",
        "inter[g,h] = sum_c count[g,c] * count[h,c] / (N_g * N_h * K)",
        "```",
        "",
        "Every token selects its channels before any group aggregation. Intra-group and inter-group "
        "overlaps are exact averages over token pairs, computed from occurrence counts without pair sampling.",
        "",
        "## Aggregate Result",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| random top-k overlap baseline | "
        f"{_mean(float(row['random_overlap_baseline']) for row in main_groups):.6f} |",
        f"| within-group token overlap | {mean_intra:.6f} |",
        f"| text-vs-SID token overlap | {mean_text_inter:.6f} |",
        f"| SID-internal token overlap | {mean_sid_inter:.6f} |",
        f"| split-half frequency cosine | "
        f"{_mean(float(row['split_frequency_cosine']) for row in main_groups):.6f} |",
        f"| recurrent-channel concentration | "
        f"{_mean(float(row['recurrent_channel_concentration']) for row in main_groups):.6f} |",
        "",
        f"Within-group overlap exceeds text-vs-SID overlap at `{intra_over_text}/{len(layer_rows)}` "
        "layer-module positions.",
        "",
        f"Within-group overlap exceeds SID-internal overlap at `{intra_over_sid}/{len(layer_rows)}` "
        "layer-module positions.",
        "",
        "## By Token Group",
        "",
        "| group | intra-token overlap | split frequency cosine | recurrent concentration |",
        "| --- | ---: | ---: | ---: |",
    ]
    for group in MAIN_TOKEN_GROUPS:
        selected = [row for row in main_groups if str(row["group"]) == group]
        lines.append(
            f"| `{group}` | "
            f"{_mean(float(row['intra_token_overlap']) for row in selected):.6f} | "
            f"{_mean(float(row['split_frequency_cosine']) for row in selected):.6f} | "
            f"{_mean(float(row['recurrent_channel_concentration']) for row in selected):.6f} |"
        )
    lines.extend(
        [
            "",
            "## By Module",
            "",
            "| module | within group | text-vs-SID | SID-internal | intra - text/SID |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for module in sorted({str(row["module"]) for row in layer_rows}, key=_module_sort_key):
        selected = [row for row in layer_rows if str(row["module"]) == module]
        lines.append(
            f"| `{module}` | "
            f"{_mean(float(row['main_group_intra_token_overlap']) for row in selected):.6f} | "
            f"{_mean(float(row['text_vs_sid_inter_token_overlap']) for row in selected):.6f} | "
            f"{_mean(float(row['sid_internal_inter_token_overlap']) for row in selected):.6f} | "
            f"{_mean(float(row['intra_minus_text_sid']) for row in selected):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Pair-Wise Token Overlap",
            "",
            "| pair | inter-token overlap | pair intra reference | intra - inter | frequency cosine |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    grouped_pairs: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped_pairs[(str(row["group_a"]), str(row["group_b"]))].append(row)
    for pair, selected in sorted(grouped_pairs.items()):
        lines.append(
            f"| `{pair[0]} - {pair[1]}` | "
            f"{_mean(float(row['inter_token_overlap']) for row in selected):.6f} | "
            f"{_mean(float(row['pair_intra_reference']) for row in selected):.6f} | "
            f"{_mean(float(row['pair_intra_minus_inter']) for row in selected):.6f} | "
            f"{_mean(float(row['frequency_cosine']) for row in selected):.6f} |"
        )
    lines.extend(
        [
            "",
            "The representative token-channel figures are qualitative. All overlap and frequency "
            "statistics use every valid calibration token.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
