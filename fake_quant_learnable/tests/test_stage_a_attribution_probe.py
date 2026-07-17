from __future__ import annotations

import torch
import torch.nn as nn

from fake_quant_learnable.modules import GPTQFakeQuantLinear
from fake_quant_learnable.quant import activation_per_token_qdq_forward
from fake_quant_learnable.stage_a_attribution_probe import stage_a_attribution_context


class _TinyStageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = GPTQFakeQuantLinear(
            weight_qdq=torch.zeros(2, 2),
            bias=None,
            act_quant="per_token",
        )
        self.linear.register_buffer("stage_probe_weight_fp", torch.eye(2), persistent=False)

    def forward(self, input_ids: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        del input_ids
        return self.linear(hidden_states)


def test_stage_a_w16_restores_only_tail_weight() -> None:
    model = _TinyStageModel()
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    input_ids = torch.ones(1, 2, dtype=torch.long)

    with stage_a_attribution_context(model, "w16"):
        output = model(input_ids=input_ids, hidden_states=hidden)

    torch.testing.assert_close(output[:, :-1], torch.zeros_like(output[:, :-1]))
    expected_activation = activation_per_token_qdq_forward(hidden)
    torch.testing.assert_close(output[:, -1], expected_activation[:, -1])
