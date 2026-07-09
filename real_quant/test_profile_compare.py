from __future__ import annotations

from pathlib import Path

from real_quant.profile_compare import (
    categorize_op,
    compare_profiles,
    render_markdown_report,
    write_profile_comparison_outputs,
)


def test_categorize_op_groups_kernel_names() -> None:
    assert categorize_op("aten::mm") == "bf16_linear_mm"
    assert categorize_op("aten::_scaled_mm") == "fp8_scaled_mm"
    assert categorize_op("_C::dynamic_per_token_scaled_fp8_quant") == "fp8_activation_quant"
    assert categorize_op("aten::scaled_dot_product_attention") == "attention"
    assert categorize_op("cudaLaunchKernel") == "kernel_launch"
    assert categorize_op("aten::copy_") == "dtype_copy"
    assert categorize_op("aten::gather") == "beam_indexing"


def test_compare_profiles_sums_categories_and_speedup() -> None:
    baseline = {
        "latency_generate_total_s": 10.0,
        "top_cuda_ops": [
            {"key": "aten::mm", "count": 4, "cuda_ms": 500.0, "cpu_ms": 20.0},
            {"key": "aten::scaled_dot_product_attention", "count": 2, "cuda_ms": 100.0, "cpu_ms": 5.0},
        ],
    }
    candidate = {
        "latency_generate_total_s": 8.0,
        "top_cuda_ops": [
            {"key": "aten::_scaled_mm", "count": 4, "cuda_ms": 250.0, "cpu_ms": 15.0},
            {"key": "_C::dynamic_per_token_scaled_fp8_quant", "count": 2, "cuda_ms": 25.0, "cpu_ms": 4.0},
        ],
    }

    comparison = compare_profiles(baseline, candidate, baseline_name="BF16", candidate_name="W8A8")

    assert comparison["latency"]["generate_speedup"] == 1.25
    by_name = {row["category"]: row for row in comparison["category_rows"]}
    assert by_name["bf16_linear_mm"]["baseline_cuda_ms"] == 500.0
    assert by_name["fp8_scaled_mm"]["candidate_cuda_ms"] == 250.0
    assert by_name["fp8_activation_quant"]["candidate_cuda_ms"] == 25.0


def test_render_markdown_report_contains_core_tables() -> None:
    comparison = {
        "baseline_name": "BF16",
        "candidate_name": "W8A8",
        "latency": {
            "baseline_generate_s": 10.0,
            "candidate_generate_s": 8.0,
            "generate_speedup": 1.25,
            "generate_reduction_pct": 20.0,
        },
        "category_rows": [
            {
                "category": "bf16_linear_mm",
                "baseline_cuda_ms": 500.0,
                "candidate_cuda_ms": 0.0,
                "delta_cuda_ms": -500.0,
                "candidate_vs_baseline_ratio": 0.0,
            }
        ],
        "top_baseline_ops": [],
        "top_candidate_ops": [],
    }

    md = render_markdown_report(comparison)

    assert "# Profiling Comparison" in md
    assert "1.250x" in md
    assert "bf16_linear_mm" in md


def test_write_profile_comparison_outputs_writes_json_and_markdown(tmp_path: Path) -> None:
    comparison = {
        "baseline_name": "BF16",
        "candidate_name": "W8A8",
        "latency": {
            "baseline_generate_s": 10.0,
            "candidate_generate_s": 8.0,
            "generate_speedup": 1.25,
            "generate_reduction_pct": 20.0,
        },
        "category_rows": [],
        "top_baseline_ops": [],
        "top_candidate_ops": [],
    }

    outputs = write_profile_comparison_outputs(comparison, tmp_path, make_plots=False)

    assert outputs["json"].exists()
    assert outputs["markdown"].exists()
    assert "png_categories" not in outputs
