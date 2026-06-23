from __future__ import annotations

from real_quant.compare_latency import build_latency_comparison, format_markdown_table


def test_build_latency_comparison_reports_speedups() -> None:
    baseline = {
        "quant_config": {"backend": "hf_full_precision"},
        "latency": {
            "num_samples": 2,
            "generate_time_total": 10.0,
            "end_to_end_time_total": 12.0,
            "generated_tokens_per_generate_second": 100.0,
            "samples_per_end_to_end_second": 0.2,
        },
    }
    candidate = {
        "quant_config": {"backend": "hf_real_naive_w8a8_scaled_mm"},
        "latency": {
            "num_samples": 2,
            "generate_time_total": 5.0,
            "end_to_end_time_total": 8.0,
            "generated_tokens_per_generate_second": 200.0,
            "samples_per_end_to_end_second": 0.25,
        },
    }

    comparison = build_latency_comparison(baseline, candidate)

    assert comparison["baseline_backend"] == "hf_full_precision"
    assert comparison["candidate_backend"] == "hf_real_naive_w8a8_scaled_mm"
    assert comparison["num_samples_baseline"] == 2
    assert comparison["num_samples_candidate"] == 2
    assert comparison["generate_time_speedup"] == 2.0
    assert comparison["end_to_end_time_speedup"] == 1.5


def test_format_markdown_table_contains_primary_latency_fields() -> None:
    comparison = build_latency_comparison(
        {
            "quant_config": {"backend": "hf_full_precision"},
            "latency": {
                "num_samples": 2,
                "generate_time_total": 10.0,
                "end_to_end_time_total": 12.0,
                "generated_tokens_per_generate_second": 100.0,
                "samples_per_end_to_end_second": 0.2,
            },
        },
        {
            "quant_config": {"backend": "hf_real_naive_w8a8_scaled_mm"},
            "latency": {
                "num_samples": 2,
                "generate_time_total": 5.0,
                "end_to_end_time_total": 8.0,
                "generated_tokens_per_generate_second": 200.0,
                "samples_per_end_to_end_second": 0.25,
            },
        },
    )

    table = format_markdown_table(comparison)

    assert "generate_time_total" in table
    assert "end_to_end_time_total" in table
    assert "2.0000x" in table
    assert "1.5000x" in table
