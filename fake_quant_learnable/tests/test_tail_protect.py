from __future__ import annotations

import torch
import torch.nn as nn

from fake_quant_learnable.tail_protect import (
    TailProtectedBaselineFakeQuantLinear,
    activation_per_token_qdq_forward_tail_protected,
)
from fake_quant_learnable.quant import activation_per_token_qdq_forward


def test_tail_protected_activation_keeps_last_token_exact() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)

    protected = activation_per_token_qdq_forward_tail_protected(
        x,
        tail_tokens=1,
        qmax=448.0,
        eps=1e-12,
    )
    regular = activation_per_token_qdq_forward(x, qmax=448.0, eps=1e-12)

    torch.testing.assert_close(protected[:, -1:, :], x[:, -1:, :], rtol=0, atol=0)
    torch.testing.assert_close(protected[:, :-1, :], regular[:, :-1, :], rtol=0, atol=0)


def test_tail_protected_linear_matches_manual_prepared_forward() -> None:
    torch.manual_seed(1)
    linear = nn.Linear(8, 4, bias=True)
    wrapper = TailProtectedBaselineFakeQuantLinear(linear, act_quant="per_token", tail_tokens=1)
    x = torch.randn(2, 3, 8)

    prepared = activation_per_token_qdq_forward_tail_protected(
        x,
        tail_tokens=1,
        qmax=wrapper.qmax,
        eps=wrapper.eps,
    )

    torch.testing.assert_close(wrapper(x), wrapper.forward_prepared(prepared), rtol=0, atol=0)
