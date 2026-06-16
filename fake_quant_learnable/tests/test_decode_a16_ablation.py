from __future__ import annotations

import torch
import torch.nn as nn

from fake_quant_learnable.quant import activation_per_token_qdq_forward
from fake_quant_learnable.support.ablation.decode_a16 import (
    DecodeA16BaselineFakeQuantLinear,
    activation_per_token_qdq_forward_decode_a16,
)


def test_decode_a16_quantizes_all_prefill_tokens() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)

    prepared = activation_per_token_qdq_forward_decode_a16(
        x,
        qmax=448.0,
        eps=1e-12,
    )
    regular = activation_per_token_qdq_forward(x, qmax=448.0, eps=1e-12)

    torch.testing.assert_close(prepared, regular, rtol=0, atol=0)


def test_decode_a16_keeps_single_token_decode_activation_exact() -> None:
    torch.manual_seed(1)
    x = torch.randn(32, 1, 8)

    prepared = activation_per_token_qdq_forward_decode_a16(
        x,
        qmax=448.0,
        eps=1e-12,
    )

    torch.testing.assert_close(prepared, x, rtol=0, atol=0)


def test_decode_a16_linear_matches_manual_prepared_forward() -> None:
    torch.manual_seed(2)
    linear = nn.Linear(8, 4, bias=True)
    wrapper = DecodeA16BaselineFakeQuantLinear(linear, act_quant="per_token")
    x = torch.randn(32, 1, 8)

    prepared = activation_per_token_qdq_forward_decode_a16(
        x,
        qmax=wrapper.qmax,
        eps=wrapper.eps,
    )

    torch.testing.assert_close(wrapper(x), wrapper.forward_prepared(prepared), rtol=0, atol=0)
