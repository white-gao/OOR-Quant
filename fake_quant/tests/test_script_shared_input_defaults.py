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
