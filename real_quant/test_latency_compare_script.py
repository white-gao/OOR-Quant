from __future__ import annotations

import subprocess
from pathlib import Path


def test_latency_compare_script_uses_matching_runners_and_compare_cli() -> None:
    script = Path("real_quant/run_naive_w8a8_latency_compare.sh")
    text = script.read_text(encoding="utf-8")

    assert "real_quant.full_precision.run_hf_baseline" in text
    assert "real_quant.naive_w8a8.run_hf_naive_w8a8" in text
    assert "real_quant.compare_latency" in text
    assert "--batch_size" in text
    assert '"${BATCH_SIZE}"' in text
    assert "BASELINE_JSON" in text
    assert "CANDIDATE_JSON" in text
    assert "ACT_QUANT_MODE" not in text
    assert "--act_quant_mode" not in text


def test_latency_compare_script_has_valid_bash_syntax() -> None:
    subprocess.run(
        ["bash", "-n", "real_quant/run_naive_w8a8_latency_compare.sh"],
        check=True,
    )
