from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from fake_quant_learnable.gradient_weights import (
    _build_teacher_forcing_batch,
    _extract_logits,
    _forward_model,
    _last_prompt_positions,
    _target_tensor,
    _teacher_forcing_full_sid_loss,
)
from fake_quant_learnable.token_weights import SLOT_TOKEN_GROUPS, build_prompt_slot_token_group_batches
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
    build_first_sid_target_token_ids,
    build_model_batches,
    build_sid_teacher_forcing_target_token_ids,
    default_calib_split,
    format_prompt,
    get_transformer_layers,
    parse_layer_indices,
)


DEFAULT_OUTPUT_ROOT = "fake_quant_learnable/results/analysis/channel_sensitivity_probe"
DEFAULT_LINEAR_REGEX = r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


@dataclass
class ChannelSums:
    grad2_sum: torch.Tensor | None = None
    gx_sum: torch.Tensor | None = None
    count: int = 0

    def add(self, *, grad2: torch.Tensor, gx: torch.Tensor, count: int) -> None:
        grad2_cpu = grad2.detach().float().cpu()
        gx_cpu = gx.detach().float().cpu()
        if self.grad2_sum is None:
            self.grad2_sum = torch.zeros_like(grad2_cpu)
            self.gx_sum = torch.zeros_like(gx_cpu)
        assert self.gx_sum is not None
        self.grad2_sum += grad2_cpu
        self.gx_sum += gx_cpu
        self.count += int(count)

    def profile(self, metric: str) -> torch.Tensor:
        if self.count <= 0:
            raise ValueError("Cannot build a channel profile with zero count.")
        if metric == "grad2":
            if self.grad2_sum is None:
                raise ValueError("Missing grad2 sums.")
            return self.grad2_sum / float(self.count)
        if metric == "gx":
            if self.gx_sum is None:
                raise ValueError("Missing gx sums.")
            return self.gx_sum / float(self.count)
        raise ValueError(f"Unsupported metric: {metric}")


@dataclass
class ChannelProbeResult:
    sums: dict[tuple[int, str, str], ChannelSums] = field(default_factory=dict)
    token_profiles: dict[tuple[int, str, str, str], list[torch.Tensor]] = field(default_factory=dict)

    def add(
        self,
        *,
        layer: int,
        module: str,
        group: str,
        grad2: torch.Tensor,
        gx: torch.Tensor,
        count: int,
    ) -> None:
        key = (int(layer), module, group)
        if key not in self.sums:
            self.sums[key] = ChannelSums()
        self.sums[key].add(grad2=grad2, gx=gx, count=count)

    def add_token_profiles(
        self,
        *,
        layer: int,
        module: str,
        metric: str,
        group: str,
        profiles: torch.Tensor,
        max_per_group: int,
    ) -> None:
        if max_per_group <= 0 or profiles.numel() == 0:
            return
        prepared = profiles.detach().float().reshape(-1, profiles.shape[-1]).cpu()
        norms = prepared.double().norm(dim=1)
        prepared = prepared[norms > 0.0]
        if prepared.numel() == 0:
            return
        key = (int(layer), module, metric, group)
        bucket = self.token_profiles.setdefault(key, [])
        remaining = int(max_per_group) - len(bucket)
        if remaining <= 0:
            return
        for row in prepared[:remaining]:
            bucket.append(row.clone())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe output-channel sensitivity profiles for slot-aware GPTQ diagnostics."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--task", default="ad", choices=RECOMMENDATION_TASKS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--calib_split", default=None)
    parser.add_argument("--sample_size", default="32")
    parser.add_argument("--layers", default="last:4", help='Layer spec: "all", "last:K", or "0,2-4".')
    parser.add_argument("--linear_regex", default=DEFAULT_LINEAR_REGEX)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt_token", default="<|sid_begin|>")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--grad_weight_loss_mode",
        choices=("first_sid", "full_sid_multi_target"),
        default="full_sid_multi_target",
    )
    parser.add_argument("--grad_weight_max_targets", type=int, default=4)
    parser.add_argument("--topk", default="1,8,32,64")
    parser.add_argument(
        "--max_token_profiles_per_group",
        type=int,
        default=32,
        help="Maximum nonzero token-level channel profiles kept for each layer/module/metric/slot group.",
    )
    parser.add_argument(
        "--max_pair_samples",
        type=int,
        default=1024,
        help="Maximum sampled token-profile pairs per intra/inter consistency statistic.",
    )
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
    batches = build_model_batches(tokenizer=tokenizer, prompts=prompts, device=device)
    slot_group_batches = build_prompt_slot_token_group_batches(
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
    )

    target_token_ids = None
    teacher_forcing_target_token_ids = None
    if args.grad_weight_loss_mode == "full_sid_multi_target":
        teacher_forcing_target_token_ids = build_sid_teacher_forcing_target_token_ids(
            tokenizer,
            samples,
            max_items=args.grad_weight_max_targets,
        )
    else:
        target_token_ids = build_first_sid_target_token_ids(tokenizer, samples)

    layers = get_transformer_layers(model)
    layer_indices = parse_layer_indices(args.layers, num_layers=len(layers))
    topk = _parse_topk(args.topk)
    output_dir = _resolve_output_dir(args, layer_indices=layer_indices, sample_count=len(samples))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[probe_channel_sensitivity] collecting channel profiles "
        f"task={args.task}, split={calib_split}, samples={len(samples)}, "
        f"layers={layer_indices}, loss_mode={args.grad_weight_loss_mode}, regex={args.linear_regex}"
    )
    result = collect_channel_sensitivity(
        model=model,
        layers=layers,
        layer_indices=layer_indices,
        model_batches=batches,
        slot_group_batches=slot_group_batches,
        linear_regex=args.linear_regex,
        target_token_ids=target_token_ids,
        teacher_forcing_target_token_ids=teacher_forcing_target_token_ids,
        max_token_profiles_per_group=args.max_token_profiles_per_group,
    )

    profile_rows, similarity_rows, consistency_rows, profile_tensors = build_probe_outputs(
        result,
        topk=topk,
        max_pair_samples=args.max_pair_samples,
    )
    _write_csv(output_dir / "channel_profile_summary.csv", profile_rows)
    _write_csv(output_dir / "channel_profile_similarity.csv", similarity_rows)
    _write_csv(output_dir / "channel_token_consistency.csv", consistency_rows)
    torch.save(profile_tensors, output_dir / "channel_profiles.pt")

    summary = {
        "task": args.task,
        "split": calib_split,
        "sample_size": len(samples),
        "layers": layer_indices,
        "linear_regex": args.linear_regex,
        "target": (
            "multi_target_full_sid_teacher_forcing"
            if args.grad_weight_loss_mode == "full_sid_multi_target"
            else "last_prompt_position_to_first_ground_truth_sid_token"
        ),
        "gradient_token_weight_loss_mode": args.grad_weight_loss_mode,
        "gradient_token_weight_max_targets": args.grad_weight_max_targets,
        "slot_groups": list(SLOT_TOKEN_GROUPS),
        "metrics": ["grad2", "gx"],
        "topk": topk,
        "max_token_profiles_per_group": args.max_token_profiles_per_group,
        "max_pair_samples": args.max_pair_samples,
        "output_files": {
            "profile_summary_csv": "channel_profile_summary.csv",
            "profile_similarity_csv": "channel_profile_similarity.csv",
            "token_consistency_csv": "channel_token_consistency.csv",
            "profile_tensors": "channel_profiles.pt",
            "report": "report.md",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "report.md").write_text(
        build_report(
            summary=summary,
            profile_rows=profile_rows,
            similarity_rows=similarity_rows,
            consistency_rows=consistency_rows,
        ),
        encoding="utf-8",
    )
    print(f"[probe_channel_sensitivity] saved to {output_dir}")


def collect_channel_sensitivity(
    *,
    model: nn.Module,
    layers: Sequence[nn.Module],
    layer_indices: Sequence[int],
    model_batches: Sequence[Mapping[str, Any]],
    slot_group_batches: Sequence[torch.Tensor],
    linear_regex: str,
    target_token_ids: Sequence[torch.Tensor | int] | None = None,
    teacher_forcing_target_token_ids: Sequence[Sequence[torch.Tensor | int]] | None = None,
    max_token_profiles_per_group: int = 32,
) -> ChannelProbeResult:
    selected = sorted(set(int(idx) for idx in layer_indices))
    if not selected:
        return ChannelProbeResult()
    use_teacher_forcing = teacher_forcing_target_token_ids is not None
    if (target_token_ids is None) == (teacher_forcing_target_token_ids is None):
        raise ValueError("Pass exactly one of target_token_ids or teacher_forcing_target_token_ids.")
    expected_targets = teacher_forcing_target_token_ids if use_teacher_forcing else target_token_ids
    if expected_targets is None:
        raise RuntimeError("Internal error: gradient targets are not initialized.")
    if len(model_batches) != len(expected_targets):
        raise ValueError("Target length does not match model batch length.")
    if len(model_batches) != len(slot_group_batches):
        raise ValueError("slot_group_batches length does not match model batch length.")

    module_pattern = re.compile(linear_regex)
    selected_set = set(selected)
    min_selected = selected[0]
    captured: dict[tuple[int, str], torch.Tensor] = {}
    handles = []
    result = ChannelProbeResult()

    def make_seed_hook(layer_idx: int):
        def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
            if layer_idx != min_selected:
                return None
            if args:
                hidden = args[0]
                if not torch.is_tensor(hidden):
                    return None
                hidden = hidden.detach().requires_grad_(True)
                return (hidden, *args[1:]), kwargs
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
            if torch.is_tensor(output):
                if not output.requires_grad:
                    return
                output.retain_grad()
                captured[(layer_idx, module_name)] = output

        return hook

    handles.append(layers[min_selected].register_forward_pre_hook(make_seed_hook(min_selected), with_kwargs=True))
    for layer_idx in selected:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of range for {len(layers)} layers.")
        for module_name, module in layers[layer_idx].named_modules():
            if isinstance(module, nn.Linear) and module_pattern.search(module_name):
                handles.append(module.register_forward_hook(make_linear_hook(layer_idx, module_name)))

    was_training = model.training
    param_requires_grad = [param.requires_grad for param in model.parameters()]
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    try:
        iterator = zip(model_batches, expected_targets, slot_group_batches)
        for batch_idx, (batch, target_info, slot_groups) in enumerate(iterator):
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
                group_ids = _teacher_forcing_prompt_group_ids(
                    slot_groups=slot_groups,
                    prompt_mask=prompt_mask,
                    prompt_len=prompt_len,
                    repeat=int(expanded_batch["input_ids"].shape[0]),
                )
                token_slice = slice(0, prompt_len)
            else:
                outputs = _forward_model(model, batch)
                logits = _extract_logits(outputs)
                target = _target_tensor(target_info, logits=logits)  # type: ignore[arg-type]
                positions = _last_prompt_positions(batch, logits=logits)
                batch_indices = torch.arange(logits.shape[0], device=logits.device)
                loss = torch.nn.functional.cross_entropy(logits[batch_indices, positions, :].float(), target)
                group_ids = slot_groups.to(device=logits.device)
                token_slice = slice(None)

            loss.backward()
            if not captured:
                raise RuntimeError(
                    "No linear outputs were captured. Check --layers and --linear_regex."
                )

            for (layer_idx, module_name), activation in captured.items():
                if layer_idx not in selected_set:
                    continue
                grad = activation.grad
                if grad is None:
                    continue
                act_prompt = activation[:, token_slice, :].detach().float()
                grad_prompt = grad[:, token_slice, :].detach().float()
                if act_prompt.shape[:2] != group_ids.shape:
                    raise ValueError(
                        f"Group shape {tuple(group_ids.shape)} does not match captured activation "
                        f"{tuple(act_prompt.shape[:2])} for layer={layer_idx}, module={module_name}, batch={batch_idx}."
                    )
                grad2 = grad_prompt.square()
                gx = (act_prompt * grad_prompt).abs()
                for group_id, group_name in enumerate(SLOT_TOKEN_GROUPS):
                    group_mask = group_ids == int(group_id)
                    if not group_mask.any():
                        continue
                    result.add(
                        layer=layer_idx,
                        module=module_name,
                        group=group_name,
                        grad2=grad2[group_mask].sum(dim=0),
                        gx=gx[group_mask].sum(dim=0),
                        count=int(group_mask.sum().item()),
                    )
                    result.add_token_profiles(
                        layer=layer_idx,
                        module=module_name,
                        metric="grad2",
                        group=group_name,
                        profiles=_token_profiles_for_group(
                            grad2,
                            group_ids,
                            group_id=group_id,
                            collapse_repeated_rows=use_teacher_forcing,
                        ),
                        max_per_group=max_token_profiles_per_group,
                    )
                    result.add_token_profiles(
                        layer=layer_idx,
                        module=module_name,
                        metric="gx",
                        group=group_name,
                        profiles=_token_profiles_for_group(
                            gx,
                            group_ids,
                            group_id=group_id,
                            collapse_repeated_rows=use_teacher_forcing,
                        ),
                        max_per_group=max_token_profiles_per_group,
                    )
            model.zero_grad(set_to_none=True)
            print(
                f"[probe_channel_sensitivity] sample {batch_idx + 1}/{len(model_batches)} "
                f"captured_modules={len(captured)}"
            )
    finally:
        for handle in handles:
            handle.remove()
        for param, requires_grad in zip(model.parameters(), param_requires_grad):
            param.requires_grad_(requires_grad)
        model.train(was_training)

    return result


def build_probe_outputs(
    result: ChannelProbeResult,
    *,
    topk: Sequence[int],
    max_pair_samples: int = 1024,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, torch.Tensor]]:
    profile_rows: list[dict[str, Any]] = []
    profiles: dict[tuple[int, str, str, str], torch.Tensor] = {}
    profile_tensors: dict[str, torch.Tensor] = {}

    for (layer, module, group), sums in sorted(result.sums.items()):
        for metric in ("grad2", "gx"):
            profile = sums.profile(metric)
            key = (layer, module, metric, group)
            profiles[key] = profile
            profile_tensors[f"layer{layer}.{module}.{metric}.{group}"] = profile
            row = {
                "layer": layer,
                "module": module,
                "group": group,
                "metric": metric,
                "count": sums.count,
            }
            row.update(channel_profile_stats(profile, topk=topk))
            profile_rows.append(row)

    similarity_rows: list[dict[str, Any]] = []
    layer_modules = sorted({(layer, module, metric) for layer, module, metric, _group in profiles})
    max_topk = max(topk) if topk else 32
    for layer, module, metric in layer_modules:
        for group_a, group_b in itertools.combinations(SLOT_TOKEN_GROUPS, 2):
            key_a = (layer, module, metric, group_a)
            key_b = (layer, module, metric, group_b)
            if key_a not in profiles or key_b not in profiles:
                continue
            profile_a = profiles[key_a]
            profile_b = profiles[key_b]
            norm_a = float(profile_a.detach().double().norm().item())
            norm_b = float(profile_b.detach().double().norm().item())
            row = {
                "layer": layer,
                "module": module,
                "metric": metric,
                "group_a": group_a,
                "group_b": group_b,
                "norm_a": _round_float(norm_a),
                "norm_b": _round_float(norm_b),
                "valid_pair": norm_a > 0.0 and norm_b > 0.0,
                "cosine": cosine_similarity(profile_a, profile_b),
            }
            for k in topk:
                row[f"top{k}_overlap"] = channel_topk_overlap(profile_a, profile_b, k=k)
            row["topmax_overlap"] = channel_topk_overlap(profile_a, profile_b, k=max_topk)
            similarity_rows.append(row)

    consistency_rows = build_token_consistency_rows(
        result.token_profiles,
        topk=topk,
        max_pairs=max_pair_samples,
    )

    return profile_rows, similarity_rows, consistency_rows, profile_tensors


def build_token_consistency_rows(
    token_profiles: Mapping[tuple[int, str, str, str], Sequence[torch.Tensor]],
    *,
    topk: Sequence[int],
    max_pairs: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    layer_modules = sorted({(layer, module, metric) for layer, module, metric, _group in token_profiles})
    for layer, module, metric in layer_modules:
        by_group = {
            group: list(token_profiles.get((layer, module, metric, group), []))
            for group in SLOT_TOKEN_GROUPS
        }
        for group in SLOT_TOKEN_GROUPS:
            other_profiles: list[torch.Tensor] = []
            for other_group, profiles in by_group.items():
                if other_group != group:
                    other_profiles.extend(profiles)
            row = {
                "layer": layer,
                "module": module,
                "metric": metric,
                "group": group,
            }
            row.update(
                token_profile_consistency_stats(
                    by_group[group],
                    other_profiles,
                    topk=topk,
                    max_pairs=max_pairs,
                )
            )
            rows.append(row)
    return rows


def token_profile_consistency_stats(
    group_profiles: Sequence[torch.Tensor],
    other_profiles: Sequence[torch.Tensor],
    *,
    topk: Sequence[int] = (32,),
    max_pairs: int = 1024,
) -> dict[str, float | int]:
    group = _valid_profile_list(group_profiles)
    other = _valid_profile_list(other_profiles)
    intra = _profile_pair_stats(group, group, same=True, topk=topk, max_pairs=max_pairs)
    inter = _profile_pair_stats(group, other, same=False, topk=topk, max_pairs=max_pairs)

    stats: dict[str, float | int] = {
        "token_profiles": len(group),
        "other_token_profiles": len(other),
        "intra_pairs": int(intra["pairs"]),
        "inter_pairs": int(inter["pairs"]),
        "intra_cosine": intra["cosine"],
        "avg_inter_cosine": inter["cosine"],
        "cosine_separation": _round_float(float(intra["cosine"]) - float(inter["cosine"])),
    }
    for k in topk:
        k_int = int(k)
        overlap_key = f"top{k_int}_overlap"
        stats[f"intra_top{k_int}_overlap"] = intra.get(overlap_key, 0.0)
        stats[f"avg_inter_top{k_int}_overlap"] = inter.get(overlap_key, 0.0)
        stats[f"top{k_int}_overlap_separation"] = _round_float(
            float(intra.get(overlap_key, 0.0)) - float(inter.get(overlap_key, 0.0))
        )
    return stats


def channel_profile_stats(profile: torch.Tensor, *, topk: Sequence[int] = (1, 8, 32, 64)) -> dict[str, float | int]:
    values = profile.detach().float().reshape(-1).clamp_min(0)
    num_channels = int(values.numel())
    if num_channels == 0:
        return {"num_channels": 0}
    total = float(values.sum().item())
    mean = float(values.mean().item())
    std = float(values.std(unbiased=False).item()) if num_channels > 1 else 0.0
    max_value = float(values.max().item())
    min_value = float(values.min().item())
    stats: dict[str, float | int] = {
        "num_channels": num_channels,
        "nonzero_channels": int((values > 0).sum().item()),
        "sum": _round_float(total),
        "l2_norm": _round_float(float(values.double().norm().item())),
        "mean": _round_float(mean),
        "std": _round_float(std),
        "cv": _round_float(std / mean) if mean > 0 else 0.0,
        "min": _round_float(min_value),
        "max": _round_float(max_value),
        "max_to_mean": _round_float(max_value / mean) if mean > 0 else 0.0,
    }
    for k in topk:
        k_int = max(1, min(int(k), num_channels))
        top_sum = float(torch.topk(values, k=k_int).values.sum().item())
        stats[f"top{k_int}_share"] = _round_float(top_sum / total) if total > 0 else 0.0
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


def channel_topk_overlap(a: torch.Tensor, b: torch.Tensor, *, k: int) -> float:
    a_flat = a.detach().float().reshape(-1)
    b_flat = b.detach().float().reshape(-1)
    if a_flat.numel() != b_flat.numel():
        raise ValueError(f"Profile shapes differ: {tuple(a_flat.shape)} vs {tuple(b_flat.shape)}")
    if a_flat.numel() == 0:
        return 0.0
    k_int = max(1, min(int(k), int(a_flat.numel())))
    a_top = set(int(idx) for idx in torch.topk(a_flat, k=k_int).indices.tolist())
    b_top = set(int(idx) for idx in torch.topk(b_flat, k=k_int).indices.tolist())
    return _round_float(len(a_top & b_top) / float(k_int))


def build_report(
    *,
    summary: Mapping[str, Any],
    profile_rows: Sequence[Mapping[str, Any]],
    similarity_rows: Sequence[Mapping[str, Any]],
    consistency_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Channel Sensitivity Probe",
        "",
        f"- task: `{summary['task']}`",
        f"- split: `{summary['split']}`",
        f"- sample_size: `{summary['sample_size']}`",
        f"- layers: `{summary['layers']}`",
        f"- linear_regex: `{summary['linear_regex']}`",
        f"- target: `{summary['target']}`",
        "",
        "## How to Read",
        "",
        "`grad2` 对齐 GuidedQuant 中 Hessian token 权重的来源；`gx` 是 `|activation * gradient|` 的辅助敏感度。"
        "slot pair similarity 比较不同 slot 的平均 channel profile；token-level consistency 比较同一 slot 内 token profile 是否比跨 slot 更相似。",
        "",
    ]
    lines.extend(_summary_table(profile_rows, similarity_rows, consistency_rows))
    return "\n".join(lines) + "\n"


def _summary_table(
    profile_rows: Sequence[Mapping[str, Any]],
    similarity_rows: Sequence[Mapping[str, Any]],
    consistency_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    rows = ["## Aggregate Signals", ""]
    profile_by_metric: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in profile_rows:
        profile_by_metric[str(row["metric"])].append(row)
    rows.append("| metric | avg cv | avg top32 share | max top32 share |")
    rows.append("| --- | ---: | ---: | ---: |")
    for metric in ("grad2", "gx"):
        metric_rows = profile_by_metric.get(metric, [])
        rows.append(
            "| {metric} | {cv:.6f} | {top32:.6f} | {max_top32:.6f} |".format(
                metric=metric,
                cv=_mean_float(row.get("cv", 0.0) for row in metric_rows),
                top32=_mean_float(row.get("top32_share", 0.0) for row in metric_rows),
                max_top32=max((float(row.get("top32_share", 0.0)) for row in metric_rows), default=0.0),
            )
        )
    rows.extend(["", "## Slot Pair Similarity", ""])
    sim_by_metric: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in similarity_rows:
        if bool(row.get("valid_pair", True)):
            sim_by_metric[str(row["metric"])].append(row)
    rows.append("| metric | avg cosine | min cosine | avg top32 overlap | min top32 overlap |")
    rows.append("| --- | ---: | ---: | ---: | ---: |")
    for metric in ("grad2", "gx"):
        metric_rows = sim_by_metric.get(metric, [])
        rows.append(
            "| {metric} | {cos:.6f} | {min_cos:.6f} | {overlap:.6f} | {min_overlap:.6f} |".format(
                metric=metric,
                cos=_mean_float(row.get("cosine", 0.0) for row in metric_rows),
                min_cos=min((float(row.get("cosine", 0.0)) for row in metric_rows), default=0.0),
                overlap=_mean_float(row.get("top32_overlap", 0.0) for row in metric_rows),
                min_overlap=min((float(row.get("top32_overlap", 0.0)) for row in metric_rows), default=0.0),
            )
        )
    rows.extend(["", "## Token-Level Slot Consistency", ""])
    consistency_by_metric: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in consistency_rows:
        if int(row.get("intra_pairs", 0)) > 0 and int(row.get("inter_pairs", 0)) > 0:
            consistency_by_metric[str(row["metric"])].append(row)
    rows.append("| metric | avg intra cosine | avg inter cosine | avg separation | avg intra top32 overlap | avg inter top32 overlap |")
    rows.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for metric in ("grad2", "gx"):
        metric_rows = consistency_by_metric.get(metric, [])
        rows.append(
            "| {metric} | {intra:.6f} | {inter:.6f} | {sep:.6f} | {intra_top32:.6f} | {inter_top32:.6f} |".format(
                metric=metric,
                intra=_mean_float(row.get("intra_cosine", 0.0) for row in metric_rows),
                inter=_mean_float(row.get("avg_inter_cosine", 0.0) for row in metric_rows),
                sep=_mean_float(row.get("cosine_separation", 0.0) for row in metric_rows),
                intra_top32=_mean_float(row.get("intra_top32_overlap", 0.0) for row in metric_rows),
                inter_top32=_mean_float(row.get("avg_inter_top32_overlap", 0.0) for row in metric_rows),
            )
        )
    rows.extend(["", "完整逐层结果见 `channel_profile_summary.csv`、`channel_profile_similarity.csv` 和 `channel_token_consistency.csv`。"])
    return rows



def _token_profiles_for_group(
    values: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    group_id: int,
    collapse_repeated_rows: bool,
) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError(f"Expected values shape [batch, seq, channel], got {tuple(values.shape)}")
    groups = group_ids.to(device=values.device).long()
    if groups.shape != values.shape[:2]:
        raise ValueError(f"Group shape {tuple(groups.shape)} does not match values {tuple(values.shape[:2])}")

    if collapse_repeated_rows and values.shape[0] > 1:
        reference = groups[0]
        if torch.equal(groups, reference.unsqueeze(0).expand_as(groups)):
            pos_mask = reference == int(group_id)
            if not pos_mask.any():
                return values.new_zeros((0, values.shape[-1]))
            return values[:, pos_mask, :].sum(dim=0)

    mask = groups == int(group_id)
    if not mask.any():
        return values.new_zeros((0, values.shape[-1]))
    return values[mask]


def _valid_profile_list(profiles: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    valid: list[torch.Tensor] = []
    for profile in profiles:
        prepared = profile.detach().float().reshape(-1)
        if prepared.numel() == 0:
            continue
        if float(prepared.double().norm().item()) == 0.0:
            continue
        valid.append(prepared)
    return valid


def _profile_pair_stats(
    profiles_a: Sequence[torch.Tensor],
    profiles_b: Sequence[torch.Tensor],
    *,
    same: bool,
    topk: Sequence[int],
    max_pairs: int,
) -> dict[str, float | int]:
    if same:
        pairs = [(i, j) for i in range(len(profiles_a)) for j in range(i + 1, len(profiles_a))]
    else:
        pairs = [(i, j) for i in range(len(profiles_a)) for j in range(len(profiles_b))]
    if max_pairs > 0 and len(pairs) > max_pairs:
        pairs = [pairs[idx] for idx in _sample_indices(len(pairs), max_pairs)]
    if not pairs:
        stats: dict[str, float | int] = {"pairs": 0, "cosine": 0.0}
        for k in topk:
            stats[f"top{int(k)}_overlap"] = 0.0
        return stats

    cosine_values: list[float] = []
    overlap_values: dict[int, list[float]] = {int(k): [] for k in topk}
    for idx_a, idx_b in pairs:
        profile_a = profiles_a[idx_a]
        profile_b = profiles_a[idx_b] if same else profiles_b[idx_b]
        cosine_values.append(cosine_similarity(profile_a, profile_b))
        for k in topk:
            k_int = int(k)
            overlap_values[k_int].append(channel_topk_overlap(profile_a, profile_b, k=k_int))

    stats = {
        "pairs": len(pairs),
        "cosine": _round_float(sum(cosine_values) / float(len(cosine_values))),
    }
    for k_int, values in overlap_values.items():
        stats[f"top{k_int}_overlap"] = _round_float(sum(values) / float(len(values))) if values else 0.0
    return stats


def _sample_indices(total: int, limit: int) -> list[int]:
    if limit <= 0 or total <= limit:
        return list(range(total))
    if limit == 1:
        return [0]
    step = (total - 1) / float(limit - 1)
    indices = [min(total - 1, int(round(i * step))) for i in range(limit)]
    deduped: list[int] = []
    seen: set[int] = set()
    for idx in indices:
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    return deduped

def _teacher_forcing_prompt_group_ids(
    *,
    slot_groups: torch.Tensor,
    prompt_mask: torch.Tensor,
    prompt_len: int,
    repeat: int,
) -> torch.Tensor:
    groups = slot_groups.to(device=prompt_mask.device).long()
    valid_groups = groups[prompt_mask.bool()].reshape(-1)
    if int(valid_groups.numel()) != int(prompt_len):
        raise ValueError(
            f"Valid slot group count {int(valid_groups.numel())} does not match prompt_len {int(prompt_len)}."
        )
    return valid_groups.unsqueeze(0).expand(repeat, -1)


def _resolve_output_dir(args: argparse.Namespace, *, layer_indices: Sequence[int], sample_count: int) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    layer_tag = "all_layers" if args.layers == "all" else f"layers_{args.layers.replace(':', '').replace(',', '_')}"
    model_tag = Path(args.model_path.rstrip("/")).name
    loss_tag = args.grad_weight_loss_mode
    if args.grad_weight_loss_mode == "full_sid_multi_target":
        loss_tag = f"fullsid_mt{args.grad_weight_max_targets}"
    return Path(DEFAULT_OUTPUT_ROOT) / f"{args.task}_{model_tag}_s{sample_count}_{layer_tag}_{loss_tag}"


def _parse_topk(spec: str) -> list[int]:
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"topk values must be positive, got {value}.")
        values.append(value)
    return sorted(set(values))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _mean_float(values: Any) -> float:
    prepared = [float(value) for value in values]
    if not prepared:
        return 0.0
    return sum(prepared) / float(len(prepared))


def _round_float(value: float) -> float:
    return round(float(value), 12)


if __name__ == "__main__":
    main()
