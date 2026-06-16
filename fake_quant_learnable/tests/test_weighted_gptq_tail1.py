from __future__ import annotations

import torch
import torch.nn as nn

from fake_quant_learnable.apply import _shared_prepare_input
from fake_quant_learnable.modules import GPTQFakeQuantLinear
from fake_quant_learnable.quant import (
    activation_per_token_qdq_forward,
    activation_per_token_qdq_forward_tail_protected,
)
from fake_quant_learnable.run_m1_onerec_ad import apply_gptq_fp8_layers, parse_args


class SingleLayerToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(4, 3)])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        del attention_mask
        return self.model.layers[0](input_ids)


def test_tail_protected_activation_keeps_prefill_last_token_and_decode_exact() -> None:
    torch.manual_seed(0)
    prefill = torch.randn(2, 4, 8)

    prepared = activation_per_token_qdq_forward_tail_protected(
        prefill,
        tail_tokens=1,
        qmax=448.0,
        eps=1e-12,
    )
    regular = activation_per_token_qdq_forward(prefill, qmax=448.0, eps=1e-12)

    torch.testing.assert_close(prepared[:, :-1, :], regular[:, :-1, :], rtol=0, atol=0)
    torch.testing.assert_close(prepared[:, -1:, :], prefill[:, -1:, :], rtol=0, atol=0)

    decode = torch.randn(32, 1, 8)
    decoded = activation_per_token_qdq_forward_tail_protected(
        decode,
        tail_tokens=1,
        qmax=448.0,
        eps=1e-12,
    )
    torch.testing.assert_close(decoded, decode, rtol=0, atol=0)


def test_gptq_fake_quant_linear_uses_tail_protected_activation() -> None:
    torch.manual_seed(1)
    linear = nn.Linear(8, 4)
    wrapper = GPTQFakeQuantLinear(
        weight_qdq=linear.weight.detach(),
        bias=linear.bias.detach(),
        act_quant="per_token",
        activation_tail_tokens=1,
    )
    x = torch.randn(2, 3, 8)

    prepared = activation_per_token_qdq_forward_tail_protected(
        x,
        tail_tokens=1,
        qmax=wrapper.qmax,
        eps=wrapper.eps,
    )

    torch.testing.assert_close(wrapper(x), wrapper.forward_prepared(prepared), rtol=0, atol=0)


def test_shared_input_prepare_uses_gptq_tail_protected_activation() -> None:
    torch.manual_seed(2)
    q_proj = GPTQFakeQuantLinear(
        weight_qdq=torch.eye(4),
        bias=None,
        act_quant="per_token",
        activation_tail_tokens=1,
    )
    k_proj = GPTQFakeQuantLinear(
        weight_qdq=torch.eye(4),
        bias=None,
        act_quant="per_token",
        activation_tail_tokens=1,
    )
    x = torch.randn(2, 4, 4)

    prepared = _shared_prepare_input((q_proj, k_proj), x)
    regular = activation_per_token_qdq_forward(x, qmax=q_proj.qmax, eps=q_proj.eps)

    assert prepared is not None
    torch.testing.assert_close(prepared[:, :-1, :], regular[:, :-1, :], rtol=0, atol=0)
    torch.testing.assert_close(prepared[:, -1:, :], x[:, -1:, :], rtol=0, atol=0)


def test_apply_gptq_layers_can_enable_tail1_activation() -> None:
    torch.manual_seed(2)
    model = SingleLayerToyModel().eval()
    batches = [{"input_ids": torch.randn(1, 3, 4), "attention_mask": torch.ones(1, 3)}]

    apply_gptq_fp8_layers(
        model=model,
        model_batches=batches,
        layer_indices=[0],
        act_quant="per_token",
        act_quant_mode="per_linear",
        damp_percent=0.01,
        block_size=2,
        activation_tail_tokens=1,
    )

    assert isinstance(model.model.layers[0], GPTQFakeQuantLinear)
    assert model.model.layers[0].activation_tail_tokens == 1


def test_parse_args_accepts_weighted_gptq_tail1_mode(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["prog", "--mode", "weighted_gptq_fp8_w8a8_tail1"])

    args = parse_args()

    assert args.mode == "weighted_gptq_fp8_w8a8_tail1"
