from __future__ import annotations

from real_quant.naive_w8a8.preflight import format_preflight_report, run_preflight_checks


def test_preflight_report_contains_required_runtime_fields_without_cuda_gemm() -> None:
    report = run_preflight_checks(device="cpu", run_cuda_gemm=False)

    assert "torch_version" in report
    assert "cuda_version" in report
    assert "has_float8_e4m3fn" in report
    assert "has_scaled_mm" in report
    assert report["cuda_gemm_checked"] is False


def test_format_preflight_report_marks_status() -> None:
    text = format_preflight_report(
        {
            "device": "cpu",
            "torch_version": "2.x",
            "cuda_version": "12.x",
            "cuda_available": False,
            "has_float8_e4m3fn": True,
            "has_scaled_mm": True,
            "cuda_gemm_checked": False,
            "cuda_gemm_ok": False,
            "status": "SKIPPED_CUDA_GEMM",
        }
    )

    assert "status: SKIPPED_CUDA_GEMM" in text
    assert "has_scaled_mm: True" in text
