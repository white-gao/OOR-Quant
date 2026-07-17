from __future__ import annotations

import torch
import torch.nn as nn

from real_quant.naive_w8a8.run_sid_stage_probe import paired_recovery, parse_variants, stages_for_variant
from real_quant.naive_w8a8.stage_rescue import (
    install_stage_rescue_model_hook,
    rescue_kind_for_input,
    stage_rescue_context,
)


class _TinyForwardModel(nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids


def test_stage_context_maps_prefill_and_two_decode_steps() -> None:
    model = _TinyForwardModel()
    handle = install_stage_rescue_model_hook(model)
    x_prefill = torch.zeros(1, 5, 4)
    x_decode = torch.zeros(1, 1, 4)
    try:
        with stage_rescue_context({"a", "c"}):
            model(input_ids=torch.ones(1, 5, dtype=torch.long))
            assert rescue_kind_for_input(x_prefill) == "tail"
            model(input_ids=torch.ones(1, 1, dtype=torch.long))
            assert rescue_kind_for_input(x_decode) is None
            model(input_ids=torch.ones(1, 1, dtype=torch.long))
            assert rescue_kind_for_input(x_decode) == "all"
    finally:
        handle.remove()


def test_parse_variants_and_stage_map() -> None:
    assert parse_variants("w8a8,rescue_a,rescue_c") == ["w8a8", "rescue_a", "rescue_c"]
    assert stages_for_variant("rescue_all") == {"a", "b", "c"}


def test_paired_recovery_uses_sid_triples() -> None:
    base = {
        "0": {"ground_truth": "<s_a_1><s_b_2><s_c_3>", "generations": ["<s_a_9><s_b_9><s_c_9>"]},
        "1": {"ground_truth": "<s_a_4><s_b_5><s_c_6>", "generations": ["<s_a_4><s_b_5><s_c_6>"]},
    }
    rescue = {
        "0": {"ground_truth": "<s_a_1><s_b_2><s_c_3>", "generations": ["<s_a_1><s_b_2><s_c_3>"]},
        "1": {"ground_truth": "<s_a_4><s_b_5><s_c_6>", "generations": ["<s_a_9><s_b_9><s_c_9>"]},
    }
    result = paired_recovery(base, rescue)
    assert result["recovery_count"] == 1
    assert result["regression_count"] == 1
    assert result["net_gain"] == 0
