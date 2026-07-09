from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


CATEGORY_ORDER = [
    "bf16_linear_mm",
    "fp8_scaled_mm",
    "fp8_activation_quant",
    "decode_w8a16_linear",
    "attention",
    "kernel_launch",
    "dtype_copy",
    "beam_indexing",
    "concat",
    "elementwise",
    "other",
]


def categorize_op(key: str) -> str:
    lowered = key.lower()
    if "decode_w8a16_linear" in lowered:
        return "decode_w8a16_linear"
    if "dynamic_per_token_scaled_fp8_quant" in lowered or "activation_dynamic_fused_quantize" in lowered:
        return "fp8_activation_quant"
    if "_scaled_mm" in lowered:
        return "fp8_scaled_mm"
    if key in {"aten::mm", "aten::matmul", "aten::linear"}:
        return "bf16_linear_mm"
    if "scaled_dot_product_attention" in lowered or "flash_attention" in lowered or "flash_fwd" in lowered:
        return "attention"
    if "cudalaunchkernel" in lowered or "culaunckernel" in lowered or "command buffer full" in lowered:
        return "kernel_launch"
    if key in {"aten::copy_", "aten::to", "aten::_to_copy"} or "copy_kernel" in lowered:
        return "dtype_copy"
    if key in {"aten::gather", "aten::index_select", "aten::topk", "aten::scatter"} or "scatter_gather" in lowered:
        return "beam_indexing"
    if key == "aten::cat" or "catarraybatchedcopy" in lowered:
        return "concat"
    if key in {"aten::mul", "aten::add", "aten::pow", "aten::mean", "aten::silu", "aten::div", "aten::clamp", "aten::abs", "aten::amax"}:
        return "elementwise"
    if "elementwise_kernel" in lowered or "vectorized_elementwise_kernel" in lowered or "reduce_kernel" in lowered:
        return "elementwise"
    return "other"


def _profile_latency_s(profile: Mapping[str, Any]) -> float:
    return float(profile.get("latency_generate_total_s", 0.0) or 0.0)


def _top_ops(profile: Mapping[str, Any], *, limit: int = 20) -> list[dict[str, Any]]:
    ops = list(profile.get("top_cuda_ops", []) or [])
    normalized: list[dict[str, Any]] = []
    for op in ops:
        normalized.append(
            {
                "key": str(op.get("key", "")),
                "category": categorize_op(str(op.get("key", ""))),
                "count": int(op.get("count", 0) or 0),
                "cuda_ms": float(op.get("cuda_ms", 0.0) or 0.0),
                "cpu_ms": float(op.get("cpu_ms", 0.0) or 0.0),
            }
        )
    normalized.sort(key=lambda row: row["cuda_ms"], reverse=True)
    return normalized[:limit]


def _sum_by_category(profile: Mapping[str, Any]) -> dict[str, dict[str, float | int]]:
    sums: dict[str, dict[str, float | int]] = defaultdict(lambda: {"cuda_ms": 0.0, "cpu_ms": 0.0, "count": 0})
    for op in profile.get("top_cuda_ops", []) or []:
        category = categorize_op(str(op.get("key", "")))
        sums[category]["cuda_ms"] = float(sums[category]["cuda_ms"]) + float(op.get("cuda_ms", 0.0) or 0.0)
        sums[category]["cpu_ms"] = float(sums[category]["cpu_ms"]) + float(op.get("cpu_ms", 0.0) or 0.0)
        sums[category]["count"] = int(sums[category]["count"]) + int(op.get("count", 0) or 0)
    return dict(sums)


def compare_profiles(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    baseline_name: str = "BF16",
    candidate_name: str = "Quant",
    top_limit: int = 20,
) -> dict[str, Any]:
    baseline_generate_s = _profile_latency_s(baseline)
    candidate_generate_s = _profile_latency_s(candidate)
    speedup = baseline_generate_s / candidate_generate_s if candidate_generate_s > 0 else 0.0
    reduction_pct = (baseline_generate_s - candidate_generate_s) / baseline_generate_s * 100.0 if baseline_generate_s > 0 else 0.0

    baseline_categories = _sum_by_category(baseline)
    candidate_categories = _sum_by_category(candidate)
    categories = sorted(set(baseline_categories) | set(candidate_categories), key=lambda item: CATEGORY_ORDER.index(item) if item in CATEGORY_ORDER else len(CATEGORY_ORDER))
    rows: list[dict[str, Any]] = []
    for category in categories:
        b_cuda = float(baseline_categories.get(category, {}).get("cuda_ms", 0.0))
        c_cuda = float(candidate_categories.get(category, {}).get("cuda_ms", 0.0))
        rows.append(
            {
                "category": category,
                "baseline_cuda_ms": b_cuda,
                "candidate_cuda_ms": c_cuda,
                "delta_cuda_ms": c_cuda - b_cuda,
                "candidate_vs_baseline_ratio": c_cuda / b_cuda if b_cuda > 0.0 else None,
                "baseline_count": int(baseline_categories.get(category, {}).get("count", 0)),
                "candidate_count": int(candidate_categories.get(category, {}).get("count", 0)),
            }
        )

    return {
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "latency": {
            "baseline_generate_s": baseline_generate_s,
            "candidate_generate_s": candidate_generate_s,
            "generate_speedup": speedup,
            "generate_reduction_pct": reduction_pct,
        },
        "category_rows": rows,
        "top_baseline_ops": _top_ops(baseline, limit=top_limit),
        "top_candidate_ops": _top_ops(candidate, limit=top_limit),
    }


def _fmt_ms(value: float) -> str:
    return f"{value:.3f}"


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}x"


def render_markdown_report(comparison: Mapping[str, Any]) -> str:
    latency = comparison["latency"]
    baseline_name = comparison["baseline_name"]
    candidate_name = comparison["candidate_name"]
    lines = [
        "# Profiling Comparison",
        "",
        "## Latency",
        "",
        "| Field | Baseline | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| Name | {baseline_name} | {candidate_name} | - |",
        f"| generate time | {latency['baseline_generate_s']:.6f}s | {latency['candidate_generate_s']:.6f}s | {latency['generate_speedup']:.3f}x speedup |",
        f"| reduction | - | - | {latency['generate_reduction_pct']:.2f}% |",
        "",
        "## CUDA Time By Category",
        "",
        "| Category | Baseline CUDA ms | Candidate CUDA ms | Candidate - Baseline ms | Ratio |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison.get("category_rows", []):
        lines.append(
            "| {category} | {b} | {c} | {d} | {r} |".format(
                category=row["category"],
                b=_fmt_ms(float(row["baseline_cuda_ms"])),
                c=_fmt_ms(float(row["candidate_cuda_ms"])),
                d=_fmt_ms(float(row["delta_cuda_ms"])),
                r=_fmt_ratio(row.get("candidate_vs_baseline_ratio")),
            )
        )

    lines.extend(["", f"## Top CUDA Ops: {baseline_name}", "", "| Op | Category | Count | CUDA ms | CPU ms |", "| --- | --- | ---: | ---: | ---: |"])
    for row in comparison.get("top_baseline_ops", []):
        lines.append(f"| `{row['key']}` | {row['category']} | {row['count']} | {_fmt_ms(float(row['cuda_ms']))} | {_fmt_ms(float(row['cpu_ms']))} |")

    lines.extend(["", f"## Top CUDA Ops: {candidate_name}", "", "| Op | Category | Count | CUDA ms | CPU ms |", "| --- | --- | ---: | ---: | ---: |"])
    for row in comparison.get("top_candidate_ops", []):
        lines.append(f"| `{row['key']}` | {row['category']} | {row['count']} | {_fmt_ms(float(row['cuda_ms']))} | {_fmt_ms(float(row['cpu_ms']))} |")

    lines.extend(
        [
            "",
            "## Reading Notes",
            "",
            "- `bf16_linear_mm` is the BF16 Linear/matmul path.",
            "- `fp8_scaled_mm` is the FP8 GEMM path used by real quant.",
            "- `fp8_activation_quant` is dynamic FP8 activation quantization overhead.",
            "- Category sums are based on profiler `top_cuda_ops`; they are diagnostic, not an exact wall-clock decomposition.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_profile_comparison_outputs(
    comparison: Mapping[str, Any],
    output_dir: str | Path,
    *,
    make_plots: bool = True,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "profile_compare.json"
    md_path = output / "profile_compare.md"
    json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown_report(comparison), encoding="utf-8")
    outputs = {"json": json_path, "markdown": md_path}
    if make_plots:
        plot_path = output / "profile_compare_categories.png"
        if write_category_plot(comparison, plot_path):
            outputs["png_categories"] = plot_path
    return outputs


def write_category_plot(comparison: Mapping[str, Any], output_path: str | Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    rows = [row for row in comparison.get("category_rows", []) if float(row.get("baseline_cuda_ms", 0.0)) > 0.0 or float(row.get("candidate_cuda_ms", 0.0)) > 0.0]
    if not rows:
        return False
    categories = [row["category"] for row in rows]
    baseline = [float(row["baseline_cuda_ms"]) for row in rows]
    candidate = [float(row["candidate_cuda_ms"]) for row in rows]
    x = list(range(len(categories)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 0.9), 4.8))
    ax.bar([v - width / 2 for v in x], baseline, width, label=str(comparison["baseline_name"]))
    ax.bar([v + width / 2 for v in x], candidate, width, label=str(comparison["candidate_name"]))
    ax.set_ylabel("CUDA time (ms)")
    ax.set_title("Profiler CUDA Time By Category")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare BF16 and quant torch profiler summaries.")
    parser.add_argument("--baseline-profile", required=True)
    parser.add_argument("--candidate-profile", required=True)
    parser.add_argument("--baseline-name", default="BF16")
    parser.add_argument("--candidate-name", default="Quant")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-limit", type=int, default=20)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = compare_profiles(
        load_json(args.baseline_profile),
        load_json(args.candidate_profile),
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
        top_limit=args.top_limit,
    )
    outputs = write_profile_comparison_outputs(comparison, args.output_dir, make_plots=not args.no_plots)
    for kind, path in outputs.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
