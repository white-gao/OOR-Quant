from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

from fake_quant_learnable.gradient_weights import (
    DEFAULT_GRADIENT_TOKEN_WEIGHT_CONFIG,
    GradientTokenWeightConfig,
    collect_gradient_token_weight_batches_by_layer,
    group_token_weight_batches_by_layer,
)
from fake_quant_learnable.token_weights import SLOT_TOKEN_GROUPS, build_prompt_slot_token_group_batches


DEFAULT_OUTPUT_ROOT = "fake_quant_learnable/results/analysis/slot_gradient_group_weight_probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe layer-wise slot-group gradient token weights for slot_grad_weighted_gptq."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--task", default="ad", choices=RECOMMENDATION_TASKS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--calib_split", default=None)
    parser.add_argument("--sample_size", default="32")
    parser.add_argument("--layers", default="all", help='Layer spec: "all", "last:K", or "0,2-4".')
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt_token", default="<|sid_begin|>")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--grad_weight_clip_percentile",
        type=float,
        default=DEFAULT_GRADIENT_TOKEN_WEIGHT_CONFIG.clip_percentile,
    )
    parser.add_argument(
        "--grad_weight_floor",
        type=float,
        default=DEFAULT_GRADIENT_TOKEN_WEIGHT_CONFIG.weight_floor,
    )
    parser.add_argument(
        "--grad_weight_normalize_mean",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GRADIENT_TOKEN_WEIGHT_CONFIG.normalize_mean,
    )
    parser.add_argument(
        "--grad_weight_loss_mode",
        choices=("first_sid", "full_sid_multi_target"),
        default="first_sid",
    )
    parser.add_argument("--grad_weight_max_targets", type=int, default=1)
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
    slot_group_batches = build_prompt_slot_token_group_batches(
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
    )

    layers = get_transformer_layers(model)
    layer_indices = parse_layer_indices(args.layers, num_layers=len(layers))
    config = GradientTokenWeightConfig(
        clip_percentile=args.grad_weight_clip_percentile,
        weight_floor=args.grad_weight_floor,
        normalize_mean=args.grad_weight_normalize_mean,
    )

    print(
        "[probe_slot_gradient_group_weights] collecting token gradients "
        f"task={args.task}, split={calib_split}, samples={len(samples)}, "
        f"layers={layer_indices}, loss_mode={args.grad_weight_loss_mode}"
    )
    token_weights_by_layer = collect_gradient_token_weight_batches_by_layer(
        model=model,
        layers=layers,
        layer_indices=layer_indices,
        model_batches=batches,
        target_token_ids=target_token_ids,
        teacher_forcing_target_token_ids=teacher_forcing_target_token_ids,
        config=config,
    )
    grouped_by_layer = group_token_weight_batches_by_layer(
        token_weights_by_layer=token_weights_by_layer,
        token_group_batches=slot_group_batches,
        model_batches=batches,
    )

    output_dir = _resolve_output_dir(args, layer_indices=layer_indices, sample_count=len(samples))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, summary_rows = build_probe_tables(
        grouped_by_layer=grouped_by_layer,
        token_group_batches=slot_group_batches,
        model_batches=batches,
        layer_indices=layer_indices,
    )
    _write_csv(output_dir / "slot_gradient_group_weights.csv", rows)
    _write_csv(output_dir / "slot_gradient_group_weight_summary.csv", summary_rows)

    summary = {
        "task": args.task,
        "split": calib_split,
        "sample_size": len(samples),
        "layers": layer_indices,
        "target": (
            "multi_target_full_sid_teacher_forcing"
            if args.grad_weight_loss_mode == "full_sid_multi_target"
            else "last_prompt_position_to_first_ground_truth_sid_token"
        ),
        "gradient_token_weight_loss_mode": args.grad_weight_loss_mode,
        "gradient_token_weight_max_targets": args.grad_weight_max_targets,
        "slot_groups": list(SLOT_TOKEN_GROUPS),
        "gradient_token_weight_config": config.to_jsonable(),
        "output_files": {
            "layer_csv": "slot_gradient_group_weights.csv",
            "summary_csv": "slot_gradient_group_weight_summary.csv",
            "report": "report.md",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "report.md").write_text(_build_report(summary, rows, summary_rows), encoding="utf-8")
    print(f"[probe_slot_gradient_group_weights] saved to {output_dir}")


def _resolve_output_dir(args: argparse.Namespace, *, layer_indices: Sequence[int], sample_count: int) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    layer_tag = "all_layers" if args.layers == "all" else f"layers_{args.layers.replace(':', '').replace(',', '_')}"
    model_tag = Path(args.model_path.rstrip("/")).name
    loss_tag = args.grad_weight_loss_mode
    if args.grad_weight_loss_mode == "full_sid_multi_target":
        loss_tag = f"fullsid_mt{args.grad_weight_max_targets}"
    return Path(DEFAULT_OUTPUT_ROOT) / f"{args.task}_{model_tag}_s{sample_count}_{layer_tag}_{loss_tag}"


def build_probe_tables(
    *,
    grouped_by_layer: Mapping[int, Sequence[torch.Tensor]],
    token_group_batches: Sequence[torch.Tensor],
    model_batches: Sequence[Mapping[str, Any]],
    layer_indices: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layer_rows: list[dict[str, Any]] = []
    by_group: dict[str, list[float]] = {group: [] for group in SLOT_TOKEN_GROUPS}
    group_counts: dict[str, int] = {group: 0 for group in SLOT_TOKEN_GROUPS}

    for layer_idx in layer_indices:
        layer_values: dict[str, list[float]] = {group: [] for group in SLOT_TOKEN_GROUPS}
        for weights, groups, batch in zip(grouped_by_layer[layer_idx], token_group_batches, model_batches):
            weight_cpu = weights.detach().float().cpu()
            group_cpu = groups.detach().long().cpu()
            mask = _valid_token_mask(batch, shape=weight_cpu.shape)
            for group_id, group_name in enumerate(SLOT_TOKEN_GROUPS):
                group_mask = mask & (group_cpu == group_id)
                if group_mask.any():
                    layer_values[group_name].extend(float(value) for value in weight_cpu[group_mask].tolist())

        text_mean = _mean(layer_values["text"])
        for group_name in SLOT_TOKEN_GROUPS:
            row = _stats_row(
                {
                    "layer": layer_idx,
                    "group": group_name,
                },
                layer_values[group_name],
                text_mean=text_mean,
            )
            layer_rows.append(row)
            if layer_values[group_name]:
                by_group[group_name].append(float(row["mean"]))
                group_counts[group_name] += int(row["count"])

    summary_text_mean = _mean(by_group["text"])
    summary_rows = [
        _stats_row(
            {
                "group": group_name,
                "token_count": group_counts[group_name],
            },
            by_group[group_name],
            text_mean=summary_text_mean,
            count_key="num_layers",
        )
        for group_name in SLOT_TOKEN_GROUPS
    ]
    return layer_rows, summary_rows


def _valid_token_mask(batch: Mapping[str, Any], *, shape: torch.Size) -> torch.Tensor:
    attention_mask = batch.get("attention_mask") if isinstance(batch, Mapping) else None
    if torch.is_tensor(attention_mask):
        mask = attention_mask.detach().bool().cpu()
        if mask.shape != shape:
            raise ValueError(f"attention_mask shape {tuple(mask.shape)} does not match weights {tuple(shape)}")
        return mask
    return torch.ones(shape, dtype=torch.bool)


def _stats_row(
    prefix: dict[str, Any],
    values: Sequence[float],
    *,
    text_mean: float,
    count_key: str = "count",
) -> dict[str, Any]:
    row = dict(prefix)
    row[count_key] = len(values)
    if not values:
        row.update({"mean": "", "median": "", "min": "", "max": "", "suggested_raw_if_text_10": ""})
        return row
    mean_value = _mean(values)
    row.update(
        {
            "mean": mean_value,
            "median": float(median(values)),
            "min": float(min(values)),
            "max": float(max(values)),
            "suggested_raw_if_text_10": mean_value / text_mean * 10.0 if text_mean > 0.0 else "",
        }
    )
    return row


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_report(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Slot Gradient Group Weight Probe",
        "",
        "设置：",
        "",
        f"- task: `{summary['task']}`",
        f"- split: `{summary['split']}`",
        f"- samples: `{summary['sample_size']}`",
        f"- layers: `{summary['layers']}`",
        f"- target: `{summary['target']}`",
        f"- groups: `{', '.join(summary['slot_groups'])}`",
        "",
        "跨层汇总：",
        "",
        "| group | token_count | num_layers | mean | median | min | max | suggested raw if text=10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {group} | {token_count} | {num_layers} | {mean} | {median} | {min} | {max} | {raw} |".format(
                group=row["group"],
                token_count=row["token_count"],
                num_layers=row["num_layers"],
                mean=_fmt(row["mean"]),
                median=_fmt(row["median"]),
                min=_fmt(row["min"]),
                max=_fmt(row["max"]),
                raw=_fmt(row["suggested_raw_if_text_10"]),
            )
        )
    lines.extend(
        [
            "",
            "逐层权重见 `slot_gradient_group_weights.csv`。",
            "",
            "前几层预览：",
            "",
            "| layer | group | count | mean | suggested raw if text=10 |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in rows[: min(50, len(rows))]:
        lines.append(
            "| {layer} | {group} | {count} | {mean} | {raw} |".format(
                layer=row["layer"],
                group=row["group"],
                count=row["count"],
                mean=_fmt(row["mean"]),
                raw=_fmt(row["suggested_raw_if_text_10"]),
            )
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value == "":
        return ""
    return f"{float(value):.6f}"


if __name__ == "__main__":
    main()
