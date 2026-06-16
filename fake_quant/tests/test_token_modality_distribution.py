from __future__ import annotations

from fake_quant.probes.activation_probe.plot_token_modality_distribution import token_modality


def test_token_modality_splits_prompt_text_and_sid_code() -> None:
    assert token_modality('<s_a_123>') == 'sid_code'
    assert token_modality('<s_b_456>') == 'sid_code'
    assert token_modality('<s_c_789>') == 'sid_code'
    assert token_modality(' item') == 'text'
    assert token_modality('<|sid_begin|>') is None
    assert token_modality('<|sid_end|>') is None
    assert token_modality('<|sid_begin|>', sid_boundary_as_sid=True) == 'sid_code'
    assert token_modality('<|sid_end|>', sid_boundary_as_sid=True) == 'sid_code'
    assert token_modality('<|im_start|>') is None
    assert token_modality('<think>') is None
