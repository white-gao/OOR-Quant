#!/usr/bin/env python3
"""
Summarize FP8 QDQ benchmark results against a baseline.

The "avg_accuracy_drop_pct" column is the mean relative drop across accuracy
metrics, excluding sample counts, timing fields, and string metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_RESULTS_DIR = Path("results/v1.0")
DEFAULT_SNAPSHOT_ROOT = Path(
    "/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots"
)
DEFAULT_OUTPUT_PREFIX = Path("results/v1.0/fp8_qdq_ad_full_summary")
BASELINE_DIR_NAME = "OneRec-1.7B_ad_full"
RESULT_PREFIX = "results_"
RESULT_SUFFIX = "_ad_full"
MODEL_NAME_PREFIX = "OneRec-1.7B-fp8e4m3-"

NON_ACCURACY_NUMERIC_FIELDS = {
    "total_samples",
    "total_time",
    "avg_time_per_sample",
}

EXPERIMENT_ORDER = [
    "baseline",
    "gate-only",
    "up-only",
    "gate-up",
    "gate-up-rerun1",
    "mlp-down",
    "mlp-all",
    "attn-o",
    "attn-qvo",
    "attn-all",
    "block-linears",
]

TOTAL_PARAMS = 2_131_878_912
FALLBACK_QUANT_STATS = {
    "gate-only": (352_321_536, 28),
    "up-only": (352_321_536, 28),
    "gate-up": (704_643_072, 56),
    "gate-up-rerun1": (704_643_072, 56),
    "mlp-down": (352_321_536, 28),
    "mlp-all": (1_056_964_608, 84),
    "attn-o": (117_440_512, 28),
    "attn-qvo": (293_601_280, 84),
    "attn-all": (352_321_536, 112),
    "block-linears": (1_409_286_144, 196),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FP8 QDQ benchmark results.")
    parser.add_argument("--results_dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--snapshot_root", default=str(DEFAULT_SNAPSHOT_ROOT))
    parser.add_argument("--output_prefix", default=str(DEFAULT_OUTPUT_PREFIX))
    parser.add_argument("--task", default="ad")
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_model_payload(eval_results: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    for key, value in eval_results.items():
        if not key.startswith("_"):
            return key, value
    raise ValueError("No model payload found in eval_results.json")


def extract_metrics(eval_path: Path, task: str, split: str) -> Tuple[str, Dict[str, Any]]:
    eval_results = load_json(eval_path)
    model_key, payload = first_model_payload(eval_results)
    try:
        metrics = payload[task][split]
    except KeyError as exc:
        raise KeyError(f"Missing {task}/{split} metrics in {eval_path}") from exc
    return model_key, metrics


def experiment_slug_from_result_dir(result_dir: Path) -> str:
    name = result_dir.name
    if name == BASELINE_DIR_NAME:
        return "baseline"

    if name.startswith(RESULT_PREFIX):
        name = name[len(RESULT_PREFIX):]
    if name.endswith(RESULT_SUFFIX):
        name = name[: -len(RESULT_SUFFIX)]
    if name.startswith(MODEL_NAME_PREFIX):
        return name[len(MODEL_NAME_PREFIX):]
    return name


def accuracy_metric_names(metrics: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for key, value in metrics.items():
        if key in NON_ACCURACY_NUMERIC_FIELDS:
            continue
        if isinstance(value, (int, float)):
            names.append(key)
    return names


def relative_drop_pct(baseline: float, value: float) -> Optional[float]:
    if baseline == 0:
        return None
    return (baseline - value) / baseline * 100.0


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def quant_report_for_slug(snapshot_root: Path, slug: str) -> Optional[Path]:
    if slug == "baseline":
        return None
    path = snapshot_root / f"{MODEL_NAME_PREFIX}{slug}" / "quant_report.json"
    return path if path.exists() else None


def load_quant_stats(snapshot_root: Path, slug: str) -> Dict[str, Any]:
    report_path = quant_report_for_slug(snapshot_root, slug)
    if report_path is None:
        if slug in FALLBACK_QUANT_STATS:
            quantized_params, quantized_tensors = FALLBACK_QUANT_STATS[slug]
            return {
                "quantized_tensors": quantized_tensors,
                "quantized_params": quantized_params,
                "total_params": TOTAL_PARAMS,
                "quantized_param_ratio": quantized_params / TOTAL_PARAMS,
                "quant_report": "",
            }
        return {
            "quantized_tensors": 0,
            "quantized_params": 0,
            "total_params": None,
            "quantized_param_ratio": 0.0,
            "quant_report": "",
        }

    report = load_json(report_path)
    return {
        "quantized_tensors": report.get("quantized_tensors"),
        "quantized_params": report.get("quantized_params"),
        "total_params": report.get("total_params"),
        "quantized_param_ratio": report.get("quantized_param_ratio"),
        "quant_report": str(report_path),
    }


def sort_key(row: Dict[str, Any]) -> Tuple[int, str]:
    slug = row["quant_setting"]
    try:
        return EXPERIMENT_ORDER.index(slug), slug
    except ValueError:
        return len(EXPERIMENT_ORDER), slug


def format_pct(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.4f}"


def format_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.8f}"


def write_markdown(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    def cell(row: Dict[str, Any], column: str) -> str:
        value = row.get(column, "")
        if isinstance(value, float):
            if column.endswith("_pct"):
                return format_pct(value)
            return format_float(value)
        return str(value)

    with path.open("w", encoding="utf-8") as f:
        f.write("# FP8 QDQ AD Full Summary\n\n")
        f.write(
            "Notes: `quantized_param_ratio_pct` is the share of model parameters that went through "
            "FP8 e4m3 QDQ. QDQ checkpoints are saved as normal floating point HF checkpoints, so this "
            "is not actual disk or GPU memory compression.\n\n"
        )
        f.write(
            "`avg_accuracy_drop_pct` is the mean relative drop versus baseline across numeric accuracy "
            "metrics only; timing and sample-count fields are excluded.\n\n"
        )
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(cell(row, col) for col in columns) + " |\n")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    snapshot_root = Path(args.snapshot_root)
    output_prefix = Path(args.output_prefix)

    baseline_eval = results_dir / BASELINE_DIR_NAME / "eval_results.json"
    if not baseline_eval.exists():
        raise FileNotFoundError(f"Baseline eval_results.json not found: {baseline_eval}")

    _, baseline_metrics = extract_metrics(baseline_eval, args.task, args.split)
    metric_names = accuracy_metric_names(baseline_metrics)

    result_paths = sorted(results_dir.glob("*/eval_results.json"))
    rows: List[Dict[str, Any]] = []

    for eval_path in result_paths:
        result_dir = eval_path.parent
        slug = experiment_slug_from_result_dir(result_dir)
        if slug == results_dir.name:
            continue

        model_key, metrics = extract_metrics(eval_path, args.task, args.split)
        if slug != "baseline" and not slug.startswith(("gate-", "up-", "mlp-", "attn-", "block-")):
            continue

        quant_stats = load_quant_stats(snapshot_root, slug)
        drops = [
            relative_drop_pct(float(baseline_metrics[name]), float(metrics[name]))
            for name in metric_names
            if name in metrics and relative_drop_pct(float(baseline_metrics[name]), float(metrics[name])) is not None
        ]

        row: Dict[str, Any] = {
            "quant_setting": slug,
            "model_key": model_key,
            "result_dir": str(result_dir),
            "quantized_param_ratio_pct": float(quant_stats["quantized_param_ratio"] or 0.0) * 100.0,
            "quantized_params": quant_stats["quantized_params"],
            "quantized_tensors": quant_stats["quantized_tensors"],
            "avg_accuracy_drop_pct": mean(drops),
        }
        for name in metric_names:
            row[name] = metrics.get(name)
            row[f"{name}_drop_pct"] = relative_drop_pct(
                float(baseline_metrics[name]), float(metrics[name])
            )
        row["total_time"] = metrics.get("total_time")
        row["avg_time_per_sample"] = metrics.get("avg_time_per_sample")
        rows.append(row)

    rows.sort(key=sort_key)

    columns = [
        "quant_setting",
        "quantized_param_ratio_pct",
        "quantized_params",
        "quantized_tensors",
        "avg_accuracy_drop_pct",
    ]
    columns.extend(metric_names)
    columns.extend(f"{name}_drop_pct" for name in metric_names)
    columns.extend(["total_time", "avg_time_per_sample", "result_dir"])

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    write_markdown(md_path, rows, columns)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote Markdown: {md_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
