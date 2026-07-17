"""Summarize outputs produced by ``run_sid_stage_probe``."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


SID_TOKEN = re.compile(r"<s_[abc]_\d+>")
SID_TRIPLE = re.compile(r"<s_a_\d+><s_b_\d+><s_c_\d+>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the SID generation stage rescue probe.")
    parser.add_argument("--probe_dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_single(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one file matching {root / pattern}, found {matches}")
    return matches[0]


def metric_block(eval_json: Mapping[str, Any]) -> Mapping[str, Any]:
    model_entries = [value for key, value in eval_json.items() if key != "_total_time"]
    if len(model_entries) != 1:
        raise ValueError("Could not identify one model entry in eval_results.json")
    task_entries = list(model_entries[0].values())
    if len(task_entries) != 1:
        raise ValueError("Could not identify one task entry in eval_results.json")
    split_entries = list(task_entries[0].values())
    if len(split_entries) != 1:
        raise ValueError("Could not identify one split entry in eval_results.json")
    return split_entries[0]


def sid_prefixes(text: str) -> list[str]:
    tokens = SID_TOKEN.findall(text)
    return ["".join(tokens[: index + 1]) for index in range(min(3, len(tokens)))]


def target_prefixes(ground_truth: str, stage_index: int) -> set[str]:
    result: set[str] = set()
    for triple in SID_TRIPLE.findall(ground_truth):
        prefixes = sid_prefixes(triple)
        if stage_index < len(prefixes):
            result.add(prefixes[stage_index])
    return result


def load_trace(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            records[str(item["sample_id"])] = item
    return records


def stage_prefix_set(record: Mapping[str, Any], stage_index: int) -> set[str]:
    result: set[str] = set()
    for beam in record.get("beams", []):
        prefixes = beam.get("prefixes", [])
        if stage_index < len(prefixes):
            result.add(str(prefixes[stage_index]))
    return result


def trajectory_summary(
    *,
    base_trace: Mapping[str, Mapping[str, Any]],
    variant_trace: Mapping[str, Mapping[str, Any]],
    samples: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common_ids = sorted(set(base_trace) & set(variant_trace) & set(samples), key=lambda value: int(value) if value.isdigit() else value)
    for stage in range(3):
        base_coverage = 0
        variant_coverage = 0
        overlaps: list[float] = []
        for sample_id in common_ids:
            target = target_prefixes(str(samples[sample_id].get("ground_truth", "")), stage)
            base_prefix = stage_prefix_set(base_trace[sample_id], stage)
            variant_prefix = stage_prefix_set(variant_trace[sample_id], stage)
            base_coverage += int(bool(target & base_prefix))
            variant_coverage += int(bool(target & variant_prefix))
            union = base_prefix | variant_prefix
            overlaps.append(float(len(base_prefix & variant_prefix)) / float(len(union)) if union else 1.0)
        count = len(common_ids)
        rows.append(
            {
                "stage": chr(ord("A") + stage),
                "samples": count,
                "w8a8_returned_prefix_coverage": base_coverage / count if count else 0.0,
                "variant_returned_prefix_coverage": variant_coverage / count if count else 0.0,
                "coverage_delta": (variant_coverage - base_coverage) / count if count else 0.0,
                "mean_returned_prefix_jaccard_vs_w8a8": mean(overlaps) if overlaps else 0.0,
            }
        )
    return rows


def fmt(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    args = parse_args()
    root = Path(args.probe_dir).resolve()
    summary = load_json(root / "stage_probe_summary.json")
    variants = ["w8a8", *summary["paired_recovery_vs_w8a8"].keys()]
    generated: dict[str, dict[str, Any]] = {}
    metrics: dict[str, Mapping[str, Any]] = {}
    traces: dict[str, dict[str, dict[str, Any]]] = {}
    for variant in variants:
        generated_path = find_single(root / variant, "**/test_generated.json")
        generated[variant] = load_json(generated_path)
        metrics[variant] = metric_block(load_json(root / variant / "eval_results.json"))
        traces[variant] = load_trace(root / f"trace_{variant}.jsonl")

    base_samples = generated["w8a8"]["samples"]
    trajectory = {
        variant: trajectory_summary(base_trace=traces["w8a8"], variant_trace=traces[variant], samples=base_samples)
        for variant in variants
        if variant != "w8a8"
    }
    report_data = {
        "metrics": {variant: dict(metrics[variant]) for variant in variants},
        "paired_recovery": summary["paired_recovery_vs_w8a8"],
        "trajectory_proxy": trajectory,
        "limitations": [
            "The rescue is W8A16: it isolates activation FP8-QDQ while retaining GPTQ FP8-QDQ weights.",
            "Returned-beam prefix coverage is derived from final returned beams and is not an exact count of every live intermediate beam.",
            "This phase cannot distinguish BF16 weight rescue; run the fake-quant attribution phase only if a stage has a stable activation-rescue gain.",
        ],
    }
    (root / "stage_probe_analysis.json").write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# SID Generation Stage Probe Report",
        "",
        "## Setup",
        "",
        "Plain GPTQ FP8 weights are calibrated once and shared by all variants. A rescue variant uses W8A16 only at its selected autoregressive stage; all other Linear calls remain W8A8.",
        "",
        "## End-to-End Metrics",
        "",
        "| Variant | Pass@1 | Pass@16 | Pass@32 | Recall@32 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for variant in variants:
        item = metrics[variant]
        lines.append(
            f"| {variant} | {fmt(float(item['pass@1']))} | {fmt(float(item['pass@16']))} | "
            f"{fmt(float(item['pass@32']))} | {fmt(float(item['recall@32']))} |"
        )
    lines.extend(["", "## Paired Recovery versus W8A8", "", "| Variant | Recovery | Regression | Net Gain |", "| --- | ---: | ---: | ---: |"])
    for variant, item in summary["paired_recovery_vs_w8a8"].items():
        lines.append(f"| {variant} | {item['recovery_count']} | {item['regression_count']} | {item['net_gain']} |")
    lines.extend(["", "## Returned-Beam Prefix Proxy", ""])
    for variant, rows in trajectory.items():
        lines.extend([f"### {variant}", "", "| Stage | W8A8 coverage | Variant coverage | Delta | Prefix Jaccard |", "| --- | ---: | ---: | ---: | ---: |"])
        for row in rows:
            lines.append(
                f"| {row['stage']} | {fmt(row['w8a8_returned_prefix_coverage'])} | "
                f"{fmt(row['variant_returned_prefix_coverage'])} | {fmt(row['coverage_delta'])} | "
                f"{fmt(row['mean_returned_prefix_jaccard_vs_w8a8'])} |"
            )
        lines.append("")
    lines.extend(["## Interpretation", ""])
    paired = summary["paired_recovery_vs_w8a8"]
    best_variant = max(paired, key=lambda name: int(paired[name]["net_gain"])) if paired else None
    if best_variant is None or int(paired[best_variant]["net_gain"]) <= 0:
        lines.append("No single-stage activation rescue shows a positive paired net gain on this probe. Do not infer that a SID stage merits special protection from this run alone.")
    else:
        lines.append(
            f"`{best_variant}` has the largest paired recovery net gain. The next justified experiment is weight-versus-activation attribution for its corresponding stage, before changing the GPTQ Hessian objective."
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend([f"- {item}" for item in report_data["limitations"]])
    (root / "SID_Generation_Stage_Probe_Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "SID_Generation_Stage_Probe_Report.md")


if __name__ == "__main__":
    main()
