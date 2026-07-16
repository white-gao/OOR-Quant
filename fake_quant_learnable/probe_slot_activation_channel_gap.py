from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
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


DEFAULT_OUTPUT_ROOT = "fake_quant_learnable/results/analysis/slot_activation_channel_gap"
DEFAULT_LINEAR_REGEX = r"(q_proj|o_proj|gate_proj|down_proj)$"
PROFILE_METRICS = ("energy", "mean_abs", "max_abs")


@dataclass
class ActivationChannelAccumulator:
    sum_abs: torch.Tensor | None = None
    sum_sq: torch.Tensor | None = None
    max_abs: torch.Tensor | None = None
    normalized_sample_profile_sum: torch.Tensor | None = None
    token_count: int = 0
    sample_count: int = 0

    def add(self, values: torch.Tensor) -> None:
        prepared = values.detach().float().reshape(-1, values.shape[-1])
        if prepared.numel() == 0 or prepared.shape[0] == 0:
            return
        abs_values = prepared.abs()
        current_abs = abs_values.sum(dim=0)
        current_sq = prepared.square().sum(dim=0)
        current_max = abs_values.amax(dim=0)
        if self.sum_abs is None:
            self.sum_abs = torch.zeros_like(current_abs)
            self.sum_sq = torch.zeros_like(current_sq)
            self.max_abs = torch.zeros_like(current_max)
            self.normalized_sample_profile_sum = torch.zeros_like(current_sq)
        assert self.sum_sq is not None
        assert self.max_abs is not None
        assert self.normalized_sample_profile_sum is not None
        self.sum_abs += current_abs
        self.sum_sq += current_sq
        self.max_abs = torch.maximum(self.max_abs, current_max)
        self.token_count += int(prepared.shape[0])

        sample_profile = current_sq / float(prepared.shape[0])
        norm = sample_profile.double().norm()
        if float(norm.item()) > 0.0:
            self.normalized_sample_profile_sum += sample_profile / norm.to(sample_profile.dtype)
            self.sample_count += 1

    def profile(self) -> dict[str, torch.Tensor | int]:
        if self.token_count <= 0 or self.sum_abs is None or self.sum_sq is None or self.max_abs is None:
            raise ValueError("Cannot build an activation channel profile with zero tokens.")
        if self.normalized_sample_profile_sum is None:
            raise ValueError("Missing normalized sample profile sum.")
        return {
            "energy": (self.sum_sq / float(self.token_count)).detach().cpu(),
            "mean_abs": (self.sum_abs / float(self.token_count)).detach().cpu(),
            "max_abs": self.max_abs.detach().cpu(),
            "normalized_sample_profile_sum": self.normalized_sample_profile_sum.detach().cpu(),
            "token_count": int(self.token_count),
            "sample_count": int(self.sample_count),
        }


@dataclass
class ActivationChannelProbeResult:
    accumulators: dict[tuple[int, str, str], ActivationChannelAccumulator] = field(default_factory=dict)

    def add(self, *, layer: int, module: str, group: str, values: torch.Tensor) -> None:
        key = (int(layer), module, group)
        if key not in self.accumulators:
            self.accumulators[key] = ActivationChannelAccumulator()
        self.accumulators[key].add(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe slot-specific input-channel activation geometry in OneRec."
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
        help='Layers used for heatmaps. "auto" selects early/middle/late from --layers.',
    )
    parser.add_argument("--linear_regex", default=DEFAULT_LINEAR_REGEX)
    parser.add_argument("--top_fractions", default="0.01,0.05")
    parser.add_argument("--heatmap_clip", type=float, default=3.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt_token", default="<|sid_begin|>")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    plot_layers = _resolve_plot_layers(args.plot_layers, selected_layers=layer_indices, num_layers=len(layers))
    top_fractions = _parse_top_fractions(args.top_fractions)
    output_dir = _resolve_output_dir(args, sample_count=len(samples))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[probe_slot_activation_channel_gap] collecting activation profiles "
        f"task={args.task}, split={calib_split}, samples={len(samples)}, "
        f"layers={layer_indices}, regex={args.linear_regex}"
    )
    result = collect_activation_channel_gap(
        model=model,
        layers=layers,
        layer_indices=layer_indices,
        model_batches=model_batches,
        slot_group_batches=slot_group_batches,
        linear_regex=args.linear_regex,
        progress_every=args.progress_every,
    )
    profile_rows, similarity_rows, consistency_rows, gap_rows, profile_tensors = build_probe_outputs(
        result,
        top_fractions=top_fractions,
    )
    _write_csv(output_dir / "channel_profile_summary.csv", profile_rows)
    _write_csv(output_dir / "channel_pair_similarity.csv", similarity_rows)
    _write_csv(output_dir / "sample_profile_consistency.csv", consistency_rows)
    _write_csv(output_dir / "layer_channel_gap_summary.csv", gap_rows)
    torch.save(profile_tensors, output_dir / "channel_profiles.pt")

    plot_files = plot_probe_figures(
        profile_tensors=profile_tensors,
        profile_rows=profile_rows,
        gap_rows=gap_rows,
        output_dir=output_dir,
        plot_layers=plot_layers,
        heatmap_clip=args.heatmap_clip,
    )
    summary = {
        "task": args.task,
        "split": calib_split,
        "sample_size": len(samples),
        "model_path": args.model_path,
        "layers": layer_indices,
        "plot_layers": plot_layers,
        "linear_regex": args.linear_regex,
        "slot_groups": list(SLOT_TOKEN_GROUPS),
        "profile_metrics": list(PROFILE_METRICS),
        "top_fractions": top_fractions,
        "profile_definition": "energy[g,c] = mean_{token in g}(activation[token,c]^2)",
        "output_files": {
            "profile_summary": "channel_profile_summary.csv",
            "pair_similarity": "channel_pair_similarity.csv",
            "sample_consistency": "sample_profile_consistency.csv",
            "layer_gap_summary": "layer_channel_gap_summary.csv",
            "profile_tensors": "channel_profiles.pt",
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
            similarity_rows=similarity_rows,
            consistency_rows=consistency_rows,
            gap_rows=gap_rows,
        ),
        encoding="utf-8",
    )
    print(f"[probe_slot_activation_channel_gap] saved to {output_dir}")


def collect_activation_channel_gap(
    *,
    model: nn.Module,
    layers: Sequence[nn.Module],
    layer_indices: Sequence[int],
    model_batches: Sequence[Mapping[str, Any]],
    slot_group_batches: Sequence[torch.Tensor],
    linear_regex: str = DEFAULT_LINEAR_REGEX,
    progress_every: int = 10,
) -> ActivationChannelProbeResult:
    if len(model_batches) != len(slot_group_batches):
        raise ValueError("slot_group_batches length does not match model_batches length.")
    selected = sorted(set(int(idx) for idx in layer_indices))
    if not selected:
        return ActivationChannelProbeResult()
    pattern = re.compile(linear_regex)
    result = ActivationChannelProbeResult()
    handles = []
    state: dict[str, torch.Tensor | int | None] = {
        "groups": None,
        "valid_mask": None,
        "captured": 0,
    }

    def make_hook(layer_idx: int, module_name: str, linear: nn.Linear):
        def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            x = args[0] if args else kwargs.get("input", kwargs.get("hidden_states"))
            groups = state["groups"]
            valid_mask = state["valid_mask"]
            if not torch.is_tensor(x) or not torch.is_tensor(groups) or not torch.is_tensor(valid_mask):
                return
            if x.shape[-1] != linear.in_features:
                raise ValueError(
                    f"Expected input dim {linear.in_features} for layer={layer_idx}, module={module_name}; "
                    f"got {tuple(x.shape)}"
                )
            x2d = x.detach().float().reshape(-1, linear.in_features)
            group_flat = groups.to(device=x.device).reshape(-1)
            valid_flat = valid_mask.to(device=x.device).reshape(-1).bool()
            if x2d.shape[0] != group_flat.numel() or x2d.shape[0] != valid_flat.numel():
                raise ValueError(
                    f"Token mask rows {group_flat.numel()} do not match activation rows {x2d.shape[0]} "
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
                    )
            state["captured"] = int(state["captured"] or 0) + 1

        return hook

    for layer_idx in selected:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of range for {len(layers)} layers.")
        for module_name, module in layers[layer_idx].named_modules():
            if isinstance(module, nn.Linear) and pattern.search(module_name):
                handles.append(module.register_forward_pre_hook(make_hook(layer_idx, module_name, module), with_kwargs=True))
    if not handles:
        raise ValueError(f"No Linear modules matched regex {linear_regex!r} in layers {selected}.")

    was_training = model.training
    model.eval()
    forward_module = model.model if hasattr(model, "model") and hasattr(model.model, "layers") else model
    try:
        with torch.no_grad():
            for batch_idx, (batch, groups) in enumerate(zip(model_batches, slot_group_batches)):
                state["groups"] = groups
                attention_mask = batch.get("attention_mask") if isinstance(batch, Mapping) else None
                state["valid_mask"] = (
                    attention_mask.bool()
                    if torch.is_tensor(attention_mask)
                    else torch.ones_like(groups, dtype=torch.bool)
                )
                state["captured"] = 0
                try:
                    forward_module(**batch, use_cache=False)
                except TypeError:
                    forward_module(**batch)
                if int(state["captured"] or 0) == 0:
                    raise RuntimeError("No Linear inputs were captured during model forward.")
                if progress_every > 0 and (
                    (batch_idx + 1) % progress_every == 0 or batch_idx + 1 == len(model_batches)
                ):
                    print(
                        f"[probe_slot_activation_channel_gap] sample {batch_idx + 1}/{len(model_batches)} "
                        f"captured_modules={state['captured']}"
                    )
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return result


def build_probe_outputs(
    result: ActivationChannelProbeResult,
    *,
    top_fractions: Sequence[float] = (0.01, 0.05),
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, torch.Tensor],
]:
    profiles: dict[tuple[int, str, str, str], torch.Tensor] = {}
    sample_sums: dict[tuple[int, str, str], tuple[torch.Tensor, int]] = {}
    profile_rows: list[dict[str, Any]] = []
    profile_tensors: dict[str, torch.Tensor] = {}

    for (layer, module, group), accumulator in sorted(result.accumulators.items()):
        data = accumulator.profile()
        token_count = int(data["token_count"])
        sample_count = int(data["sample_count"])
        sample_sums[(layer, module, group)] = (
            data["normalized_sample_profile_sum"],  # type: ignore[arg-type]
            sample_count,
        )
        for metric in PROFILE_METRICS:
            profile = data[metric]
            assert torch.is_tensor(profile)
            profiles[(layer, module, metric, group)] = profile
            profile_tensors[f"layer{layer}.{module}.{metric}.{group}"] = profile
            row = {
                "layer": layer,
                "module": module,
                "group": group,
                "metric": metric,
                "token_count": token_count,
                "sample_count": sample_count,
            }
            row.update(channel_profile_stats(profile, top_fractions=top_fractions))
            profile_rows.append(row)

    similarity_rows: list[dict[str, Any]] = []
    layer_modules = sorted({(layer, module, metric) for layer, module, metric, _group in profiles})
    for layer, module, metric in layer_modules:
        for group_a, group_b in itertools.combinations(SLOT_TOKEN_GROUPS, 2):
            key_a = (layer, module, metric, group_a)
            key_b = (layer, module, metric, group_b)
            if key_a not in profiles or key_b not in profiles:
                continue
            a = profiles[key_a]
            b = profiles[key_b]
            cosine = cosine_similarity(a, b)
            row: dict[str, Any] = {
                "layer": layer,
                "module": module,
                "metric": metric,
                "group_a": group_a,
                "group_b": group_b,
                "cosine": cosine,
                "cosine_distance": _round_float(1.0 - cosine),
            }
            for fraction in top_fractions:
                row[f"{_fraction_tag(fraction)}_overlap"] = channel_top_fraction_overlap(
                    a,
                    b,
                    fraction=fraction,
                )
            similarity_rows.append(row)

    consistency_rows = build_sample_consistency_rows(sample_sums)
    gap_rows = build_layer_gap_rows(similarity_rows, top_fractions=top_fractions)
    return profile_rows, similarity_rows, consistency_rows, gap_rows, profile_tensors


def build_sample_consistency_rows(
    sample_sums: Mapping[tuple[int, str, str], tuple[torch.Tensor, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    layer_modules = sorted({(layer, module) for layer, module, _group in sample_sums})
    for layer, module in layer_modules:
        for group in SLOT_TOKEN_GROUPS:
            key = (layer, module, group)
            if key not in sample_sums:
                continue
            profile_sum, sample_count = sample_sums[key]
            intra = average_intra_sample_cosine(profile_sum, sample_count=sample_count)
            inter_values: list[float] = []
            for other in SLOT_TOKEN_GROUPS:
                other_key = (layer, module, other)
                if other == group or other_key not in sample_sums:
                    continue
                other_sum, other_count = sample_sums[other_key]
                inter_values.append(
                    average_inter_sample_cosine(
                        profile_sum,
                        sample_count,
                        other_sum,
                        other_count,
                    )
                )
            avg_inter = _mean(inter_values)
            rows.append(
                {
                    "layer": layer,
                    "module": module,
                    "group": group,
                    "sample_count": sample_count,
                    "intra_cosine": intra,
                    "avg_inter_cosine": avg_inter,
                    "cosine_separation": _round_float(intra - avg_inter),
                }
            )
    return rows


def build_layer_gap_rows(
    similarity_rows: Sequence[Mapping[str, Any]],
    *,
    top_fractions: Sequence[float],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in similarity_rows:
        grouped[(int(row["layer"]), str(row["module"]), str(row["metric"]))].append(row)
    rows: list[dict[str, Any]] = []
    sid_groups = {"sid_a", "sid_b", "sid_c"}
    for (layer, module, metric), pair_rows in sorted(grouped.items()):
        categories = {
            "all": pair_rows,
            "text_vs_sid": [
                row
                for row in pair_rows
                if {str(row["group_a"]), str(row["group_b"])} & {"text"}
                and ({str(row["group_a"]), str(row["group_b"])} & sid_groups)
            ],
            "sid_internal": [
                row
                for row in pair_rows
                if str(row["group_a"]) in sid_groups and str(row["group_b"]) in sid_groups
            ],
        }
        for category, selected in categories.items():
            if not selected:
                continue
            row: dict[str, Any] = {
                "layer": layer,
                "module": module,
                "metric": metric,
                "category": category,
                "pair_count": len(selected),
                "mean_cosine": _mean(float(item["cosine"]) for item in selected),
                "mean_cosine_distance": _mean(float(item["cosine_distance"]) for item in selected),
            }
            for fraction in top_fractions:
                key = f"{_fraction_tag(fraction)}_overlap"
                row[f"mean_{key}"] = _mean(float(item[key]) for item in selected)
            rows.append(row)
    return rows


def channel_profile_stats(
    profile: torch.Tensor,
    *,
    top_fractions: Sequence[float] = (0.01, 0.05),
) -> dict[str, float | int]:
    values = profile.detach().float().reshape(-1).clamp_min(0.0)
    if values.numel() == 0:
        return {"num_channels": 0}
    mean = float(values.mean().item())
    std = float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0
    total = float(values.sum().item())
    stats: dict[str, float | int] = {
        "num_channels": int(values.numel()),
        "mean": _round_float(mean),
        "std": _round_float(std),
        "cv": _round_float(std / mean) if mean > 0.0 else 0.0,
        "max": _round_float(float(values.max().item())),
        "max_to_mean": _round_float(float(values.max().item()) / mean) if mean > 0.0 else 0.0,
    }
    for fraction in top_fractions:
        k = _fraction_count(values.numel(), fraction)
        top_sum = float(torch.topk(values, k=k).values.sum().item())
        stats[f"{_fraction_tag(fraction)}_share"] = _round_float(top_sum / total) if total > 0.0 else 0.0
    return stats


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.detach().double().reshape(-1)
    b_flat = b.detach().double().reshape(-1)
    if a_flat.numel() != b_flat.numel():
        raise ValueError(f"Profile shapes differ: {tuple(a_flat.shape)} vs {tuple(b_flat.shape)}")
    denom = float(a_flat.norm().item() * b_flat.norm().item())
    if denom == 0.0:
        return 0.0
    return _round_float(float(torch.dot(a_flat, b_flat).item()) / denom)


def channel_top_fraction_overlap(a: torch.Tensor, b: torch.Tensor, *, fraction: float) -> float:
    a_flat = a.detach().float().reshape(-1)
    b_flat = b.detach().float().reshape(-1)
    if a_flat.numel() != b_flat.numel():
        raise ValueError(f"Profile shapes differ: {tuple(a_flat.shape)} vs {tuple(b_flat.shape)}")
    if a_flat.numel() == 0:
        return 0.0
    k = _fraction_count(a_flat.numel(), fraction)
    a_top = set(int(index) for index in torch.topk(a_flat, k=k).indices.tolist())
    b_top = set(int(index) for index in torch.topk(b_flat, k=k).indices.tolist())
    return _round_float(len(a_top & b_top) / float(k))


def average_intra_sample_cosine(profile_sum: torch.Tensor, *, sample_count: int) -> float:
    if sample_count < 2:
        return 0.0
    squared_norm = float(profile_sum.detach().double().square().sum().item())
    value = (squared_norm - float(sample_count)) / float(sample_count * (sample_count - 1))
    return _round_float(max(-1.0, min(1.0, value)))


def average_inter_sample_cosine(
    profile_sum_a: torch.Tensor,
    sample_count_a: int,
    profile_sum_b: torch.Tensor,
    sample_count_b: int,
) -> float:
    if sample_count_a <= 0 or sample_count_b <= 0:
        return 0.0
    value = float(
        torch.dot(
            profile_sum_a.detach().double().reshape(-1),
            profile_sum_b.detach().double().reshape(-1),
        ).item()
    ) / float(sample_count_a * sample_count_b)
    return _round_float(max(-1.0, min(1.0, value)))


def plot_probe_figures(
    *,
    profile_tensors: Mapping[str, torch.Tensor],
    profile_rows: Sequence[Mapping[str, Any]],
    gap_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    plot_layers: Sequence[int],
    heatmap_clip: float,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[probe_slot_activation_channel_gap] matplotlib unavailable; skipping plots")
        return []

    profiles = _unpack_profile_tensors(profile_tensors)
    files: list[str] = []
    for layer in plot_layers:
        modules = sorted(
            {module for candidate_layer, module, metric, _group in profiles if candidate_layer == layer and metric == "energy"},
            key=_module_sort_key,
        )
        if not modules:
            continue
        fig, axes = plt.subplots(1, len(modules), figsize=(5.0 * len(modules), 3.4), squeeze=False)
        image = None
        for axis, module in zip(axes[0], modules):
            keys = [(layer, module, "energy", group) for group in SLOT_TOKEN_GROUPS]
            if not all(key in profiles for key in keys):
                axis.set_visible(False)
                continue
            matrix = torch.stack([profiles[key].float() for key in keys])
            text_profile = matrix[SLOT_TOKEN_GROUPS.index("text")]
            relative = torch.log2((matrix + 1e-12) / (text_profile.unsqueeze(0) + 1e-12))
            order = _channel_display_order(relative)
            display = relative[:, order].clamp(-float(heatmap_clip), float(heatmap_clip)).numpy()
            image = axis.imshow(
                display,
                aspect="auto",
                interpolation="nearest",
                cmap="coolwarm",
                vmin=-float(heatmap_clip),
                vmax=float(heatmap_clip),
            )
            axis.set_title(_module_display_name(module), fontsize=10)
            axis.set_xlabel("input channels (group-sorted)")
            axis.set_yticks(range(len(SLOT_TOKEN_GROUPS)))
            axis.set_yticklabels(SLOT_TOKEN_GROUPS)
        fig.suptitle(f"Layer {layer}: log2 slot channel energy / text channel energy", fontsize=12)
        if image is not None:
            fig.colorbar(image, ax=list(axes[0]), shrink=0.75, label="log2 ratio to text")
        path = output_dir / f"slot_channel_energy_heatmap_layer_{layer:02d}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        files.append(path.name)

    energy_rows = [row for row in gap_rows if row["metric"] == "energy"]
    modules = sorted({str(row["module"]) for row in energy_rows}, key=_module_sort_key)
    if modules:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
        for axis, module in zip(axes.reshape(-1), modules):
            for category, label in (("text_vs_sid", "text vs SID"), ("sid_internal", "SID a/b/c")):
                selected = sorted(
                    [row for row in energy_rows if row["module"] == module and row["category"] == category],
                    key=lambda row: int(row["layer"]),
                )
                if selected:
                    axis.plot(
                        [int(row["layer"]) for row in selected],
                        [float(row["mean_cosine_distance"]) for row in selected],
                        marker="o",
                        markersize=3,
                        linewidth=1.5,
                        label=label,
                    )
            axis.set_title(_module_display_name(module))
            axis.set_xlabel("layer")
            axis.set_ylabel("channel-profile cosine distance")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        for axis in axes.reshape(-1)[len(modules) :]:
            axis.set_visible(False)
        fig.suptitle("Layer-wise slot activation channel gap", fontsize=12)
        path = output_dir / "layerwise_slot_channel_gap.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        files.append(path.name)

    energy_profile_rows = [row for row in profile_rows if row["metric"] == "energy"]
    modules = sorted({str(row["module"]) for row in energy_profile_rows}, key=_module_sort_key)
    if modules:
        rows = int(math.ceil(len(modules) / 2.0))
        fig, axes = plt.subplots(rows, 2, figsize=(11, 3.6 * rows), squeeze=False)
        for axis, module in zip(axes.reshape(-1), modules):
            for group in SLOT_TOKEN_GROUPS:
                selected = sorted(
                    [
                        row
                        for row in energy_profile_rows
                        if row["module"] == module and row["group"] == group
                    ],
                    key=lambda row: int(row["layer"]),
                )
                if selected:
                    axis.plot(
                        [int(row["layer"]) for row in selected],
                        [float(row["mean"]) for row in selected],
                        marker="o",
                        markersize=2.5,
                        linewidth=1.2,
                        label=group,
                    )
            axis.set_title(_module_display_name(module))
            axis.set_xlabel("layer")
            axis.set_ylabel("mean diag(H_group) / token count")
            axis.set_yscale("log")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7, ncol=2)
        for axis in axes.reshape(-1)[len(modules) :]:
            axis.set_visible(False)
        fig.suptitle("Layer-wise slot channel energy", fontsize=12)
        path = output_dir / "layerwise_group_channel_energy.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        files.append(path.name)
    return files


def build_report(
    *,
    summary: Mapping[str, Any],
    similarity_rows: Sequence[Mapping[str, Any]],
    consistency_rows: Sequence[Mapping[str, Any]],
    gap_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Slot Activation Channel Gap Probe",
        "",
        f"- task: `{summary['task']}`",
        f"- split: `{summary['split']}`",
        f"- sample_size: `{summary['sample_size']}`",
        f"- layers: `{summary['layers']}`",
        f"- linear_regex: `{summary['linear_regex']}`",
        f"- groups: `{summary['slot_groups']}`",
        "",
        "## Definition",
        "",
        "```text",
        "energy[g,c] = mean_{token in group g}(activation[token,c]^2)",
        "```",
        "",
        "`energy[g]` equals the token-count-normalized diagonal of the group Hessian `X_g.T @ X_g`.",
        "",
        "## Aggregate Channel Gap",
        "",
        "| metric | category | mean cosine distance | mean top1% overlap |",
        "| --- | --- | ---: | ---: |",
    ]
    for metric in PROFILE_METRICS:
        for category in ("text_vs_sid", "sid_internal"):
            selected = [row for row in gap_rows if row["metric"] == metric and row["category"] == category]
            lines.append(
                f"| {metric} | {category} | "
                f"{_mean(float(row['mean_cosine_distance']) for row in selected):.6f} | "
                f"{_mean(float(row.get('mean_top1pct_overlap', 0.0)) for row in selected):.6f} |"
            )

    valid_consistency = [row for row in consistency_rows if int(row["sample_count"]) >= 2]
    lines.extend(
        [
            "",
            "## Cross-Sample Stability",
            "",
            "| avg intra cosine | avg inter cosine | avg separation |",
            "| ---: | ---: | ---: |",
            "| {intra:.6f} | {inter:.6f} | {separation:.6f} |".format(
                intra=_mean(float(row["intra_cosine"]) for row in valid_consistency),
                inter=_mean(float(row["avg_inter_cosine"]) for row in valid_consistency),
                separation=_mean(float(row["cosine_separation"]) for row in valid_consistency),
            ),
            "",
            "## Strongest Energy Gaps",
            "",
            "| layer | module | category | cosine distance | top1% overlap |",
            "| ---: | --- | --- | ---: | ---: |",
        ]
    )
    strongest = sorted(
        [row for row in gap_rows if row["metric"] == "energy" and row["category"] != "all"],
        key=lambda row: float(row["mean_cosine_distance"]),
        reverse=True,
    )[:12]
    for row in strongest:
        lines.append(
            f"| {row['layer']} | {row['module']} | {row['category']} | "
            f"{float(row['mean_cosine_distance']):.6f} | "
            f"{float(row.get('mean_top1pct_overlap', 0.0)):.6f} |"
        )
    lines.extend(
        [
            "",
            "Complete per-module results are stored in the CSV files and `channel_profiles.pt`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _unpack_profile_tensors(
    tensors: Mapping[str, torch.Tensor],
) -> dict[tuple[int, str, str, str], torch.Tensor]:
    unpacked: dict[tuple[int, str, str, str], torch.Tensor] = {}
    pattern = re.compile(r"^layer(?P<layer>\d+)\.(?P<module>.+)\.(?P<metric>energy|mean_abs|max_abs)\.(?P<group>[^.]+)$")
    for name, tensor in tensors.items():
        match = pattern.match(name)
        if match is None:
            continue
        unpacked[(int(match.group("layer")), match.group("module"), match.group("metric"), match.group("group"))] = tensor
    return unpacked


def _channel_display_order(relative: torch.Tensor) -> torch.Tensor:
    centered = relative - relative.mean(dim=0, keepdim=True)
    dominant = centered.argmax(dim=0)
    contrast = centered.amax(dim=0) - centered.amin(dim=0)
    score = dominant.to(torch.float64) * (float(relative.shape[1]) + 1.0) - contrast.to(torch.float64)
    return torch.argsort(score)


def _module_sort_key(module: str) -> tuple[int, str]:
    order = {
        "self_attn.q_proj": 0,
        "self_attn.o_proj": 1,
        "mlp.gate_proj": 2,
        "mlp.down_proj": 3,
    }
    return order.get(module, 99), module


def _module_display_name(module: str) -> str:
    return {
        "self_attn.q_proj": "attention q/k/v input",
        "self_attn.o_proj": "attention output input",
        "mlp.gate_proj": "FFN gate/up input",
        "mlp.down_proj": "FFN down input",
    }.get(module, module)


def _resolve_plot_layers(spec: str, *, selected_layers: Sequence[int], num_layers: int) -> list[int]:
    if not selected_layers:
        return []
    if spec == "auto":
        candidates = [selected_layers[0], selected_layers[len(selected_layers) // 2], selected_layers[-1]]
        return list(dict.fromkeys(candidates))
    requested = parse_layer_indices(spec, num_layers=num_layers)
    selected_set = set(selected_layers)
    return [layer for layer in requested if layer in selected_set]


def _resolve_output_dir(args: argparse.Namespace, *, sample_count: int) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    model_tag = Path(args.model_path.rstrip("/")).name
    layer_tag = "all_layers" if args.layers == "all" else f"layers_{args.layers.replace(':', '').replace(',', '_')}"
    return Path(DEFAULT_OUTPUT_ROOT) / f"{args.task}_{model_tag}_s{sample_count}_{layer_tag}"


def _parse_top_fractions(spec: str) -> list[float]:
    values = sorted(set(float(part.strip()) for part in spec.split(",") if part.strip()))
    if not values or any(value <= 0.0 or value > 1.0 for value in values):
        raise ValueError(f"top_fractions must contain values in (0, 1], got {spec!r}")
    return values


def _fraction_count(num_channels: int, fraction: float) -> int:
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    return max(1, min(int(num_channels), int(math.ceil(float(num_channels) * float(fraction)))))


def _fraction_tag(fraction: float) -> str:
    percent = float(fraction) * 100.0
    if percent.is_integer():
        return f"top{int(percent)}pct"
    return f"top{str(percent).replace('.', 'p')}pct"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Sequence[float] | Any) -> float:
    prepared = list(values)
    return sum(prepared) / float(len(prepared)) if prepared else 0.0


def _round_float(value: float) -> float:
    return round(float(value), 9)


if __name__ == "__main__":
    main()
