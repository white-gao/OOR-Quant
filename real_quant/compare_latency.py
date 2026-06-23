from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


PRIMARY_FIELDS = (
    "generate_time_total",
    "end_to_end_time_total",
    "generated_tokens_per_generate_second",
    "samples_per_end_to_end_second",
)


def _float(payload: Mapping[str, Any], key: str) -> float:
    return float(payload.get(key, 0.0))


def _speedup(baseline: float, candidate: float) -> float:
    return baseline / candidate if candidate > 0 else 0.0


def _backend(payload: Mapping[str, Any]) -> str:
    config = payload.get("quant_config", {})
    if isinstance(config, Mapping):
        return str(config.get("backend", "unknown"))
    return "unknown"


def build_latency_comparison(
    baseline_payload: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
) -> dict[str, float | int | str]:
    baseline_latency = baseline_payload.get("latency", {})
    candidate_latency = candidate_payload.get("latency", {})
    if not isinstance(baseline_latency, Mapping) or not isinstance(candidate_latency, Mapping):
        raise ValueError("Both payloads must contain a top-level latency object.")

    comparison: dict[str, float | int | str] = {
        "baseline_backend": _backend(baseline_payload),
        "candidate_backend": _backend(candidate_payload),
        "num_samples_baseline": int(baseline_latency.get("num_samples", 0)),
        "num_samples_candidate": int(candidate_latency.get("num_samples", 0)),
    }
    for field in PRIMARY_FIELDS:
        baseline_value = _float(baseline_latency, field)
        candidate_value = _float(candidate_latency, field)
        comparison[f"{field}_baseline"] = baseline_value
        comparison[f"{field}_candidate"] = candidate_value
        if field.endswith("_total"):
            comparison[f"{field.removesuffix('_total')}_speedup"] = _speedup(
                baseline_value,
                candidate_value,
            )
        else:
            comparison[f"{field}_ratio"] = _speedup(candidate_value, baseline_value)
    return comparison


def load_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(value: float | int | str) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _fmt_speedup(value: float | int | str) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.4f}x"
    return str(value)


def format_markdown_table(comparison: Mapping[str, float | int | str]) -> str:
    rows = [
        (
            "generate_time_total",
            comparison["generate_time_total_baseline"],
            comparison["generate_time_total_candidate"],
            _fmt_speedup(comparison["generate_time_speedup"]),
        ),
        (
            "end_to_end_time_total",
            comparison["end_to_end_time_total_baseline"],
            comparison["end_to_end_time_total_candidate"],
            _fmt_speedup(comparison["end_to_end_time_speedup"]),
        ),
        (
            "generated_tokens_per_generate_second",
            comparison["generated_tokens_per_generate_second_baseline"],
            comparison["generated_tokens_per_generate_second_candidate"],
            _fmt_speedup(comparison["generated_tokens_per_generate_second_ratio"]),
        ),
        (
            "samples_per_end_to_end_second",
            comparison["samples_per_end_to_end_second_baseline"],
            comparison["samples_per_end_to_end_second_candidate"],
            _fmt_speedup(comparison["samples_per_end_to_end_second_ratio"]),
        ),
    ]
    lines = [
        f"baseline_backend: {comparison['baseline_backend']}",
        f"candidate_backend: {comparison['candidate_backend']}",
        f"num_samples: {comparison['num_samples_baseline']} vs {comparison['num_samples_candidate']}",
        "",
        "| field | baseline | candidate | speedup/ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    for field, baseline, candidate, ratio in rows:
        lines.append(f"| {field} | {_fmt(baseline)} | {_fmt(candidate)} | {ratio} |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare latency fields from two real_quant result JSON files.")
    parser.add_argument("--baseline", required=True, help="Full-precision generated JSON path.")
    parser.add_argument("--candidate", required=True, help="Naive W8A8 generated JSON path.")
    parser.add_argument("--json", action="store_true", help="Print raw comparison JSON instead of a markdown table.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = build_latency_comparison(load_payload(args.baseline), load_payload(args.candidate))
    if args.json:
        print(json.dumps(comparison, indent=2, ensure_ascii=False))
    else:
        print(format_markdown_table(comparison))


if __name__ == "__main__":
    main()
