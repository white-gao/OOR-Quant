from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fake_quant_learnable.probe_channel_sensitivity import (
    DEFAULT_LINEAR_REGEX,
    ChannelProbeResult,
    collect_channel_sensitivity,
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


DEFAULT_OUTPUT_ROOT = "fake_quant_learnable/results/analysis/linear_slot_weight_stability_probe"


@dataclass(frozen=True)
class LinearGroupProfile:
    profile: torch.Tensor | None
    energies: Mapping[str, float]
    counts: Mapping[str, int]

    @property
    def missing_groups(self) -> tuple[str, ...]:
        return tuple(group for group in SLOT_TOKEN_GROUPS if int(self.counts.get(group, 0)) <= 0)

    @property
    def valid(self) -> bool:
        return self.profile is not None and not self.missing_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe split-half stability of linear-wise slot-group output-gradient weights."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--task", default="ad", choices=RECOMMENDATION_TASKS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--calib_split", default=None)
    parser.add_argument("--sample_size", default="128")
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
    if len(samples) < 2:
        raise ValueError("Split-half stability requires at least two calibration samples.")

    prompts = [format_prompt(sample["prompt"], args.prompt_token) for sample in samples]
    batches = build_model_batches(tokenizer=tokenizer, prompts=prompts, device=device)
    slot_group_batches = build_prompt_slot_token_group_batches(
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
    )
    first_targets: Sequence[torch.Tensor | int] | None = None
    full_sid_targets: Sequence[Sequence[torch.Tensor | int]] | None = None
    if args.grad_weight_loss_mode == "full_sid_multi_target":
        full_sid_targets = build_sid_teacher_forcing_target_token_ids(
            tokenizer,
            samples,
            max_items=args.grad_weight_max_targets,
        )
    else:
        first_targets = build_first_sid_target_token_ids(tokenizer, samples)

    layers = get_transformer_layers(model)
    layer_indices = parse_layer_indices(args.layers, num_layers=len(layers))
    output_dir = _resolve_output_dir(args, layer_indices=layer_indices, sample_count=len(samples))
    output_dir.mkdir(parents=True, exist_ok=True)

    split_indices = {"half_a": list(range(0, len(samples), 2)), "half_b": list(range(1, len(samples), 2))}
    print(
        "[probe_linear_slot_weight_stability] collecting split-half Linear-output gradients "
        f"task={args.task}, split={calib_split}, samples={len(samples)}, "
        f"layers={layer_indices}, loss_mode={args.grad_weight_loss_mode}"
    )
    profiles_by_half: dict[str, dict[tuple[int, str], LinearGroupProfile]] = {}
    for half_name, indices in split_indices.items():
        print(
            f"[probe_linear_slot_weight_stability] {half_name}: "
            f"samples={len(indices)}, source_indices={indices[:3]}..."
        )
        result = collect_channel_sensitivity(
            model=model,
            layers=layers,
            layer_indices=layer_indices,
            model_batches=_select(batches, indices),
            slot_group_batches=_select(slot_group_batches, indices),
            linear_regex=args.linear_regex,
            target_token_ids=_select(first_targets, indices) if first_targets is not None else None,
            teacher_forcing_target_token_ids=(
                _select(full_sid_targets, indices) if full_sid_targets is not None else None
            ),
            max_token_profiles_per_group=0,
        )
        profiles_by_half[half_name] = extract_linear_group_profiles(result)

    profile_rows = build_profile_rows(profiles_by_half)
    stability_rows = build_stability_rows(profiles_by_half["half_a"], profiles_by_half["half_b"])
    module_summary_rows = summarize_stability_rows(stability_rows)
    summary = {
        "task": args.task,
        "split": calib_split,
        "sample_size": len(samples),
        "split_strategy": "alternating_input_order",
        "half_sizes": {name: len(indices) for name, indices in split_indices.items()},
        "layers": layer_indices,
        "linear_regex": args.linear_regex,
        "sensitivity": "mean_t ||dL/dY_{layer,linear,t}||_2^2, normalized across slot groups",
        "gradient_token_weight_loss_mode": args.grad_weight_loss_mode,
        "gradient_token_weight_max_targets": args.grad_weight_max_targets,
        "slot_groups": list(SLOT_TOKEN_GROUPS),
        "output_files": {
            "profiles_csv": "linear_slot_group_profiles.csv",
            "stability_csv": "linear_slot_group_stability.csv",
            "module_summary_csv": "linear_slot_group_stability_by_module.csv",
            "report": "report.md",
        },
    }
    _write_csv(output_dir / "linear_slot_group_profiles.csv", profile_rows)
    _write_csv(output_dir / "linear_slot_group_stability.csv", stability_rows)
    _write_csv(output_dir / "linear_slot_group_stability_by_module.csv", module_summary_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "report.md").write_text(
        build_report(summary=summary, stability_rows=stability_rows, module_summary_rows=module_summary_rows),
        encoding="utf-8",
    )
    print(f"[probe_linear_slot_weight_stability] saved to {output_dir}")


def normalize_group_energies(energies: Mapping[str, float], *, eps: float = 1e-12) -> torch.Tensor:
    values = torch.tensor([max(0.0, float(energies.get(group, 0.0))) for group in SLOT_TOKEN_GROUPS])
    total = float(values.sum().item())
    if total <= eps:
        raise ValueError("At least one slot-group energy must be positive.")
    return values / total


def compare_group_profiles(first: torch.Tensor, second: torch.Tensor) -> dict[str, float | bool | str]:
    a = _normalized_profile(first)
    b = _normalized_profile(second)
    if a.shape != b.shape or a.numel() != len(SLOT_TOKEN_GROUPS):
        raise ValueError(f"Expected matching {len(SLOT_TOKEN_GROUPS)}-group profiles, got {tuple(a.shape)} and {tuple(b.shape)}")

    denom = float(a.double().norm().item() * b.double().norm().item())
    cosine = 0.0 if denom == 0.0 else float(torch.dot(a.double(), b.double()).item()) / denom
    top1_a = int(torch.argmax(a).item())
    top1_b = int(torch.argmax(b).item())
    top2_a = set(int(index) for index in torch.topk(a, k=2).indices.tolist())
    top2_b = set(int(index) for index in torch.topk(b, k=2).indices.tolist())
    return {
        "cosine": _round(cosine),
        "l1": _round(float(torch.abs(a - b).sum().item())),
        "spearman": _round(_spearman(a, b)),
        "top1_group_a": SLOT_TOKEN_GROUPS[top1_a],
        "top1_group_b": SLOT_TOKEN_GROUPS[top1_b],
        "top1_agree": top1_a == top1_b,
        "top2_overlap": _round(len(top2_a & top2_b) / 2.0),
    }


def extract_linear_group_profiles(result: ChannelProbeResult) -> dict[tuple[int, str], LinearGroupProfile]:
    module_keys = sorted({(int(layer), module) for layer, module, _group in result.sums})
    profiles: dict[tuple[int, str], LinearGroupProfile] = {}
    for layer, module in module_keys:
        energies: dict[str, float] = {}
        counts: dict[str, int] = {}
        for group in SLOT_TOKEN_GROUPS:
            sums = result.sums.get((layer, module, group))
            if sums is None or sums.count <= 0:
                energies[group] = 0.0
                counts[group] = 0
                continue
            energies[group] = float(sums.profile("grad2").sum().item())
            counts[group] = int(sums.count)
        try:
            profile = normalize_group_energies(energies)
        except ValueError:
            profile = None
        profiles[(layer, module)] = LinearGroupProfile(profile=profile, energies=energies, counts=counts)
    return profiles


def build_profile_rows(
    profiles_by_half: Mapping[str, Mapping[tuple[int, str], LinearGroupProfile]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for half_name, profiles in sorted(profiles_by_half.items()):
        for (layer, module), item in sorted(profiles.items()):
            for index, group in enumerate(SLOT_TOKEN_GROUPS):
                rows.append(
                    {
                        "half": half_name,
                        "layer": layer,
                        "module": module,
                        "module_type": module.rsplit(".", 1)[-1],
                        "group": group,
                        "token_count": int(item.counts.get(group, 0)),
                        "mean_grad2_energy": _round(float(item.energies.get(group, 0.0))),
                        "normalized_weight": (
                            _round(float(item.profile[index].item())) if item.profile is not None else ""
                        ),
                    }
                )
    return rows


def build_stability_rows(
    first_profiles: Mapping[tuple[int, str], LinearGroupProfile],
    second_profiles: Mapping[tuple[int, str], LinearGroupProfile],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer, module in sorted(set(first_profiles) | set(second_profiles)):
        first = first_profiles.get((layer, module))
        second = second_profiles.get((layer, module))
        row: dict[str, Any] = {
            "layer": layer,
            "module": module,
            "module_type": module.rsplit(".", 1)[-1],
            "valid_for_stability": False,
            "missing_groups_half_a": "",
            "missing_groups_half_b": "",
        }
        if first is None or second is None:
            row["missing_groups_half_a"] = "missing_linear" if first is None else ",".join(first.missing_groups)
            row["missing_groups_half_b"] = "missing_linear" if second is None else ",".join(second.missing_groups)
            rows.append(row)
            continue
        row["missing_groups_half_a"] = ",".join(first.missing_groups)
        row["missing_groups_half_b"] = ",".join(second.missing_groups)
        if not first.valid or not second.valid:
            rows.append(row)
            continue
        assert first.profile is not None and second.profile is not None
        row["valid_for_stability"] = True
        row.update(compare_group_profiles(first.profile, second.profile))
        rows.append(row)
    return rows


def summarize_stability_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    valid_rows = [row for row in rows if bool(row.get("valid_for_stability", False))]
    by_module: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        by_module[str(row["module_type"])].append(row)

    summary_rows = [_stability_summary_row("all", valid_rows, total_modules=len(rows))]
    for module_type in sorted(by_module):
        summary_rows.append(
            _stability_summary_row(module_type, by_module[module_type], total_modules=len(by_module[module_type]))
        )
    return summary_rows


def build_report(
    *,
    summary: Mapping[str, Any],
    stability_rows: Sequence[Mapping[str, Any]],
    module_summary_rows: Sequence[Mapping[str, Any]],
) -> str:
    valid_rows = [row for row in stability_rows if bool(row.get("valid_for_stability", False))]
    invalid_count = len(stability_rows) - len(valid_rows)
    lines = [
        "# Linear-Wise Slot Weight Split-Half Stability Probe",
        "",
        f"- task: `{summary['task']}`",
        f"- split: `{summary['split']}`",
        f"- sample_size: `{summary['sample_size']}`; halves: `{summary['half_sizes']}`",
        f"- layers: `{summary['layers']}`",
        f"- sensitivity: `{summary['sensitivity']}`",
        "",
        "The probe estimates a separate normalized slot-group weight vector for every Linear in two disjoint calibration halves. "
        "It evaluates estimator stability, not quantization accuracy.",
        "",
        "## Aggregate Stability",
        "",
        f"- valid Linears: `{len(valid_rows)}/{len(stability_rows)}`; invalid due to missing group coverage: `{invalid_count}`",
        "",
        "| module type | Linears | mean cosine | median cosine | min cosine | mean Spearman | top-1 agreement | mean top-2 overlap | mean L1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in module_summary_rows:
        lines.append(
            "| {module} | {count} | {mean_cosine:.6f} | {median_cosine:.6f} | {min_cosine:.6f} | "
            "{mean_spearman:.6f} | {top1_agreement:.6f} | {mean_top2_overlap:.6f} | {mean_l1:.6f} |".format(
                module=row["module_type"],
                count=row["valid_linears"],
                mean_cosine=float(row["mean_cosine"]),
                median_cosine=float(row["median_cosine"]),
                min_cosine=float(row["min_cosine"]),
                mean_spearman=float(row["mean_spearman"]),
                top1_agreement=float(row["top1_agreement"]),
                mean_top2_overlap=float(row["mean_top2_overlap"]),
                mean_l1=float(row["mean_l1"]),
            )
        )

    lines.extend(["", "## Least Stable Linears", ""])
    lines.append("| layer | Linear | cosine | Spearman | top-1 groups | L1 |")
    lines.append("| ---: | --- | ---: | ---: | --- | ---: |")
    for row in sorted(valid_rows, key=lambda item: float(item["cosine"]))[:10]:
        lines.append(
            "| {layer} | {module} | {cosine:.6f} | {spearman:.6f} | {a} / {b} | {l1:.6f} |".format(
                layer=row["layer"],
                module=row["module"],
                cosine=float(row["cosine"]),
                spearman=float(row["spearman"]),
                a=row["top1_group_a"],
                b=row["top1_group_b"],
                l1=float(row["l1"]),
            )
        )

    lines.extend(
        [
            "",
            "A high average similarity supports testing Linear-wise weighting; it does not establish that finer weighting improves GPTAQ. "
            "The quantization ablation must still compare Linear-wise and Layer-wise weighting under the same activation-aware target and alpha.",
        ]
    )
    return "\n".join(lines) + "\n"


def _stability_summary_row(
    module_type: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    total_modules: int,
) -> dict[str, Any]:
    if not rows:
        return {
            "module_type": module_type,
            "valid_linears": 0,
            "total_linears": total_modules,
            "mean_cosine": 0.0,
            "median_cosine": 0.0,
            "min_cosine": 0.0,
            "mean_spearman": 0.0,
            "top1_agreement": 0.0,
            "mean_top2_overlap": 0.0,
            "mean_l1": 0.0,
        }
    return {
        "module_type": module_type,
        "valid_linears": len(rows),
        "total_linears": total_modules,
        "mean_cosine": _round(mean(float(row["cosine"]) for row in rows)),
        "median_cosine": _round(median(float(row["cosine"]) for row in rows)),
        "min_cosine": _round(min(float(row["cosine"]) for row in rows)),
        "mean_spearman": _round(mean(float(row["spearman"]) for row in rows)),
        "top1_agreement": _round(mean(float(bool(row["top1_agree"])) for row in rows)),
        "mean_top2_overlap": _round(mean(float(row["top2_overlap"]) for row in rows)),
        "mean_l1": _round(mean(float(row["l1"]) for row in rows)),
    }


def _normalized_profile(profile: torch.Tensor) -> torch.Tensor:
    values = profile.detach().float().reshape(-1).clamp_min(0.0)
    if values.numel() != len(SLOT_TOKEN_GROUPS):
        raise ValueError(f"Expected {len(SLOT_TOKEN_GROUPS)} group values, got {values.numel()}")
    total = values.sum().clamp_min(1e-12)
    return values / total


def _spearman(first: torch.Tensor, second: torch.Tensor) -> float:
    first_rank = _average_ranks(first)
    second_rank = _average_ranks(second)
    first_centered = first_rank - first_rank.mean()
    second_centered = second_rank - second_rank.mean()
    denom = float(first_centered.double().norm().item() * second_centered.double().norm().item())
    if denom == 0.0:
        return 0.0
    return float(torch.dot(first_centered.double(), second_centered.double()).item()) / denom


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    flattened = values.detach().float().reshape(-1)
    sorted_values, order = torch.sort(flattened, descending=True, stable=True)
    ranks = torch.empty_like(flattened)
    start = 0
    while start < sorted_values.numel():
        end = start + 1
        while end < sorted_values.numel() and bool(torch.isclose(sorted_values[end], sorted_values[start])):
            end += 1
        average_rank = (float(start + 1) + float(end)) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _select(values: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    return [values[index] for index in indices]


def _resolve_output_dir(args: argparse.Namespace, *, layer_indices: Sequence[int], sample_count: int) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    model_tag = Path(args.model_path.rstrip("/")).name
    layer_tag = "all_layers" if args.layers == "all" else f"layers_{args.layers.replace(':', '').replace(',', '_')}"
    loss_tag = args.grad_weight_loss_mode
    if args.grad_weight_loss_mode == "full_sid_multi_target":
        loss_tag = f"fullsid_mt{args.grad_weight_max_targets}"
    del layer_indices
    return Path(DEFAULT_OUTPUT_ROOT) / f"{args.task}_{model_tag}_s{sample_count}_{layer_tag}_{loss_tag}"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _round(value: float) -> float:
    return round(float(value), 9)


if __name__ == "__main__":
    main()
