from __future__ import annotations

import torch
import torch.nn as nn

from fake_quant_learnable.gptq import collect_gptq_hessians
from fake_quant_learnable.token_weights import (
    SLOT_TOKEN_GROUP_IDS,
    PromptTokenWeightConfig,
    SlotTokenWeightConfig,
    build_prompt_slot_token_group_batches,
    build_prompt_slot_token_weight_batches,
    build_prompt_token_weight_batches,
)
from fake_quant_learnable.run_m1_onerec_ad import parse_args


class CharOffsetTokenizer:
    def __call__(self, text: str, *, return_tensors: str = "pt", return_offsets_mapping: bool = False):
        assert return_tensors == "pt"
        input_ids = torch.arange(len(text), dtype=torch.long).unsqueeze(0)
        payload = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
        if return_offsets_mapping:
            offsets = torch.tensor([[(idx, idx + 1) for idx in range(len(text))]], dtype=torch.long)
            payload["offset_mapping"] = offsets
        return payload


def test_collect_gptq_hessians_supports_token_weighted_inputs() -> None:
    linear = nn.Linear(2, 2, bias=False)
    x = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    token_weights = torch.tensor([[1.0, 3.0, 2.0]])

    hessians = collect_gptq_hessians(linear, [x], token_weight_batches=[token_weights])

    x2d = x.reshape(-1, 2)
    w = token_weights.reshape(-1)
    expected = (x2d * w[:, None]).t().matmul(x2d) / w.sum()
    torch.testing.assert_close(hessians[""], expected)


def test_prompt_token_weights_distinguish_text_history_sid_interest_sid_and_boundaries() -> None:
    prompt = (
        "任务文本 "
        "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|> "
        "中间说明 "
        "<|sid_begin|><s_a_4><s_b_5><s_c_6><|sid_end|> "
        "<|sid_begin|>"
    )
    config = PromptTokenWeightConfig(
        history_sid_weight=1.0,
        interest_sid_weight=5.0,
        text_weight=10.0,
        sid_boundary_weight=2.0,
        normalize_mean=False,
    )

    weights = build_prompt_token_weight_batches(
        tokenizer=CharOffsetTokenizer(),
        prompts=[prompt],
        device=torch.device("cpu"),
        config=config,
    )[0]

    assert weights.shape == (1, len(prompt))
    assert weights[0, prompt.index("任")].item() == 10.0
    assert weights[0, prompt.index("<s_a_1>") + 1].item() == 1.0
    assert weights[0, prompt.index("<s_a_4>") + 1].item() == 5.0
    assert weights[0, prompt.index("<|sid_begin|>") + 1].item() == 2.0
    assert weights[0, prompt.rindex("<|sid_begin|>") + 1].item() == 2.0


def test_prompt_slot_token_weights_distinguish_sid_slots_and_boundaries() -> None:
    prompt = (
        "任务文本 "
        "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|> "
        "<|sid_begin|>"
    )
    config = SlotTokenWeightConfig(
        text_weight=10.0,
        sid_a_weight=5.0,
        sid_b_weight=2.0,
        sid_c_weight=2.0,
        boundary_weight=2.0,
        normalize_mean=False,
    )

    weights = build_prompt_slot_token_weight_batches(
        tokenizer=CharOffsetTokenizer(),
        prompts=[prompt],
        device=torch.device("cpu"),
        config=config,
    )[0]

    assert weights.shape == (1, len(prompt))
    assert weights[0, prompt.index("任")].item() == 10.0
    assert weights[0, prompt.index("<s_a_1>") + 1].item() == 5.0
    assert weights[0, prompt.index("<s_b_2>") + 1].item() == 2.0
    assert weights[0, prompt.index("<s_c_3>") + 1].item() == 2.0
    assert weights[0, prompt.index("<|sid_begin|>") + 1].item() == 2.0
    assert weights[0, prompt.rindex("<|sid_begin|>") + 1].item() == 2.0


def test_prompt_slot_token_group_batches_distinguish_sid_slots_and_boundaries() -> None:
    prompt = (
        "任务文本 "
        "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|> "
        "<|sid_begin|>"
    )

    groups = build_prompt_slot_token_group_batches(
        tokenizer=CharOffsetTokenizer(),
        prompts=[prompt],
        device=torch.device("cpu"),
    )[0]

    assert groups.shape == (1, len(prompt))
    assert groups[0, prompt.index("任")].item() == SLOT_TOKEN_GROUP_IDS["text"]
    assert groups[0, prompt.index("<s_a_1>") + 1].item() == SLOT_TOKEN_GROUP_IDS["sid_a"]
    assert groups[0, prompt.index("<s_b_2>") + 1].item() == SLOT_TOKEN_GROUP_IDS["sid_b"]
    assert groups[0, prompt.index("<s_c_3>") + 1].item() == SLOT_TOKEN_GROUP_IDS["sid_c"]
    assert groups[0, prompt.index("<|sid_begin|>") + 1].item() == SLOT_TOKEN_GROUP_IDS["boundary"]
    assert groups[0, prompt.rindex("<|sid_begin|>") + 1].item() == SLOT_TOKEN_GROUP_IDS["boundary"]


def test_parse_args_accepts_weighted_gptq_mode(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["prog", "--mode", "weighted_gptq_fp8_w8a8"])

    args = parse_args()

    assert args.mode == "weighted_gptq_fp8_w8a8"
    assert args.token_weight_history_sid == 1.0
    assert args.token_weight_interest_sid == 5.0
    assert args.token_weight_text == 10.0
    assert args.token_weight_sid_boundary == 2.0
    assert args.token_weight_normalize_mean is True
