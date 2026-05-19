from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_quant_run_scripts_pin_shared_input_without_env_override() -> None:
    script_paths = [
        PROJECT_ROOT / "fake_quant" / "run_hf_ad_full_quant.sh",
        PROJECT_ROOT / "fake_quant" / "smoothquant" / "run_smoothquant_ad.sh",
        PROJECT_ROOT / "fake_quant" / "ranking_margin" / "run_ranking_margin_smoothquant_ad.sh",
    ]

    for script_path in script_paths:
        text = script_path.read_text(encoding="utf-8")
        assert "ACT_QUANT_MODE" not in text
        assert "--act_quant_mode shared_input" in text


def test_run_ad_sid_keeps_per_linear_interface() -> None:
    text = (PROJECT_ROOT / "fake_quant" / "run_ad_sid.py").read_text(encoding="utf-8")

    assert 'choices=["per_linear", "shared_input"]' in text


def test_smoothquant_script_exposes_smooth_layer_cutoff() -> None:
    script_text = (PROJECT_ROOT / "fake_quant" / "smoothquant" / "run_smoothquant_ad.sh").read_text(
        encoding="utf-8"
    )
    runner_text = (PROJECT_ROOT / "fake_quant" / "run_ad_sid.py").read_text(encoding="utf-8")

    assert "SMOOTH_LAYER_MIN" in script_text
    assert "SMOOTH_LAYER_CUTOFF" in script_text
    assert "--smooth_layer_min" in script_text
    assert "--smooth_layer_cutoff" in script_text
    assert "--smooth_layer_min" in runner_text
    assert "--smooth_layer_cutoff" in runner_text


def test_quant_run_scripts_expose_generation_and_calibration_batch_size() -> None:
    script_paths = [
        PROJECT_ROOT / "fake_quant" / "run_hf_ad_full_quant.sh",
        PROJECT_ROOT / "fake_quant" / "smoothquant" / "run_smoothquant_ad.sh",
        PROJECT_ROOT / "fake_quant" / "ranking_margin" / "run_ranking_margin_smoothquant_ad.sh",
    ]

    for script_path in script_paths:
        text = script_path.read_text(encoding="utf-8")
        assert "BATCH_SIZE" in text
        assert "CALIB_BATCH_SIZE" in text
        assert "--batch_size" in text
