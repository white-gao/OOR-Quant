from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from fake_quant_learnable.gradient_weights import (
    GradientTokenWeightConfig,
    collect_gradient_group_token_weight_batches_by_layer,
    collect_gradient_token_weight_batches_by_layer,
    normalize_gradient_token_weights,
)
from fake_quant_learnable.linear_gradient_weights import (
    collect_linear_gradient_group_token_weight_batches_by_layer,
)
from fake_quant_learnable.run_m1_onerec_ad import parse_args
from fake_quant_learnable.token_weights import (
    PROMPT_TOKEN_GROUP_IDS,
    build_prompt_token_group_batches,
)


class ToyGradBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2, bias=False), nn.Linear(2, 3, bias=False)])


class ToyGradModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = ToyGradBackbone()
        with torch.no_grad():
            self.model.layers[0].weight.copy_(torch.tensor([[1.0, 0.5], [-0.25, 1.0]]))
            self.model.layers[1].weight.copy_(
                torch.tensor(
                    [
                        [0.2, -0.3],
                        [1.0, 0.4],
                        [-0.4, 0.9],
                    ]
                )
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
    ):
        del attention_mask, use_cache
        hidden = input_ids.float()
        hidden = self.model.layers[0](hidden)
        hidden = hidden + hidden.mean(dim=1, keepdim=True)
        hidden = self.model.layers[1](hidden)
        return SimpleNamespace(logits=hidden)


class ToyTokenGradModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 4)
        self.model = SimpleNamespace()
        self.model.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.lm_head = nn.Linear(4, 16, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
    ):
        del attention_mask, use_cache
        hidden = self.embed(input_ids.long())
        for layer in self.model.layers:
            hidden = torch.tanh(layer(hidden))
        return SimpleNamespace(logits=self.lm_head(hidden))


def test_normalize_gradient_token_weights_clips_floors_and_normalizes_mean() -> None:
    sensitivity = torch.tensor([[0.0, 1.0, 100.0, 2.0]])
    config = GradientTokenWeightConfig(
        clip_percentile=50.0,
        weight_floor=0.1,
        normalize_mean=True,
    )

    weights = normalize_gradient_token_weights(sensitivity, attention_mask=None, config=config)

    assert weights.shape == sensitivity.shape
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0), rtol=1e-6, atol=1e-6)
    assert torch.all(weights > 0)
    assert weights[0, 2] < 10.0


def test_collect_gradient_token_weights_uses_hidden_times_grad_sensitivity() -> None:
    torch.manual_seed(0)
    model = ToyGradModel().eval()
    batch = {
        "input_ids": torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }

    weights_by_layer = collect_gradient_token_weight_batches_by_layer(
        model=model,
        layers=model.model.layers,
        layer_indices=[0, 1],
        model_batches=[batch],
        target_token_ids=[torch.tensor(1)],
        config=GradientTokenWeightConfig(clip_percentile=100.0, weight_floor=0.0, normalize_mean=True),
    )

    assert set(weights_by_layer) == {0, 1}
    layer0_weights = weights_by_layer[0][0]
    layer1_weights = weights_by_layer[1][0]
    assert layer0_weights.shape == (1, 3)
    assert layer1_weights.shape == (1, 3)
    torch.testing.assert_close(layer0_weights.mean(), torch.tensor(1.0), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(layer1_weights.mean(), torch.tensor(1.0), rtol=1e-5, atol=1e-5)
    assert not torch.allclose(layer0_weights, torch.ones_like(layer0_weights))
    assert not torch.allclose(layer0_weights, layer1_weights)


def test_collect_gradient_token_weights_supports_full_sid_multi_target_loss() -> None:
    torch.manual_seed(0)
    model = ToyTokenGradModel().eval()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }

    weights_by_layer = collect_gradient_token_weight_batches_by_layer(
        model=model,
        layers=model.model.layers,
        layer_indices=[0, 1],
        model_batches=[batch],
        teacher_forcing_target_token_ids=[
            [
                torch.tensor([4, 5, 6]),
                torch.tensor([7, 8, 9]),
            ]
        ],
        config=GradientTokenWeightConfig(clip_percentile=100.0, weight_floor=0.0, normalize_mean=True),
    )

    assert set(weights_by_layer) == {0, 1}
    for layer_idx in [0, 1]:
        weights = weights_by_layer[layer_idx][0]
        assert weights.shape == (1, 3)
        torch.testing.assert_close(weights.mean(), torch.tensor(1.0), rtol=1e-5, atol=1e-5)
        assert not torch.allclose(weights, torch.ones_like(weights))


def test_collect_linear_gradient_group_weights_supports_full_sid_multi_target() -> None:
    torch.manual_seed(0)
    model = ToyTokenGradModel().eval()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }

    weights = collect_linear_gradient_group_token_weight_batches_by_layer(
        model=model,
        layers=model.model.layers,
        layer_indices=[0, 1],
        model_batches=[batch],
        teacher_forcing_target_token_ids=[[torch.tensor([4, 5]), torch.tensor([6, 7])]],
        token_group_batches=[torch.tensor([[0, 1, 0]])],
        linear_regex=r"$",
        config=GradientTokenWeightConfig(clip_percentile=100.0, weight_floor=0.0, normalize_mean=True),
    )

    for layer_idx in [0, 1]:
        grouped = weights[layer_idx][""][0]
        assert grouped.shape == (1, 3)
        torch.testing.assert_close(grouped[:, 0], grouped[:, 2])
        assert torch.isfinite(grouped).all()


def test_collect_gradient_group_weights_replaces_tokens_with_layer_group_means() -> None:
    torch.manual_seed(0)
    model = ToyGradModel().eval()
    batches = [
        {
            "input_ids": torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[[0.5, 1.0], [1.5, -0.5], [-1.0, 2.0]]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        },
    ]
    token_group_batches = [
        torch.tensor(
            [[
                PROMPT_TOKEN_GROUP_IDS["text"],
                PROMPT_TOKEN_GROUP_IDS["history_sid"],
                PROMPT_TOKEN_GROUP_IDS["text"],
            ]]
        ),
        torch.tensor(
            [[
                PROMPT_TOKEN_GROUP_IDS["history_sid"],
                PROMPT_TOKEN_GROUP_IDS["interest_sid"],
                PROMPT_TOKEN_GROUP_IDS["text"],
            ]]
        ),
    ]
    config = GradientTokenWeightConfig(clip_percentile=100.0, weight_floor=0.0, normalize_mean=True)

    token_weights_by_layer = collect_gradient_token_weight_batches_by_layer(
        model=model,
        layers=model.model.layers,
        layer_indices=[0, 1],
        model_batches=batches,
        target_token_ids=[torch.tensor(1), torch.tensor(2)],
        config=config,
    )
    group_weights_by_layer = collect_gradient_group_token_weight_batches_by_layer(
        model=model,
        layers=model.model.layers,
        layer_indices=[0, 1],
        model_batches=batches,
        target_token_ids=[torch.tensor(1), torch.tensor(2)],
        token_group_batches=token_group_batches,
        config=config,
    )

    for layer_idx, token_weight_batches in token_weights_by_layer.items():
        group_sums: dict[int, float] = {}
        group_counts: dict[int, int] = {}
        for weights, groups in zip(token_weight_batches, token_group_batches):
            for group_id in torch.unique(groups).tolist():
                mask = groups == int(group_id)
                group_sums[int(group_id)] = group_sums.get(int(group_id), 0.0) + float(weights[mask].sum())
                group_counts[int(group_id)] = group_counts.get(int(group_id), 0) + int(mask.sum())
        group_means = {group_id: group_sums[group_id] / group_counts[group_id] for group_id in group_sums}

        for grouped_weights, groups in zip(group_weights_by_layer[layer_idx], token_group_batches):
            expected = torch.zeros_like(grouped_weights)
            for group_id, mean in group_means.items():
                expected = torch.where(groups == group_id, torch.full_like(expected, mean), expected)
            torch.testing.assert_close(grouped_weights, expected)


def test_collect_linear_gradient_group_weights_returns_linear_specific_group_smoothing() -> None:
    torch.manual_seed(0)
    model = ToyGradModel().eval()
    batches = [
        {
            "input_ids": torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
    ]
    groups = [torch.tensor([[0, 1, 0]])]

    weights = collect_linear_gradient_group_token_weight_batches_by_layer(
        model=model,
        layers=model.model.layers,
        layer_indices=[0, 1],
        model_batches=batches,
        target_token_ids=[torch.tensor(1)],
        token_group_batches=groups,
        linear_regex=r"$",
        config=GradientTokenWeightConfig(clip_percentile=100.0, weight_floor=0.0, normalize_mean=True),
    )

    assert set(weights) == {0, 1}
    assert set(weights[0]) == {""}
    assert set(weights[1]) == {""}
    for layer_idx in [0, 1]:
        grouped = weights[layer_idx][""][0]
        assert grouped.shape == (1, 3)
        torch.testing.assert_close(grouped[:, 0], grouped[:, 2])
        assert float(grouped[:, 1].item()) > 0.0


def test_prompt_token_group_batches_distinguish_roles() -> None:
    from fake_quant_learnable.tests.test_weighted_gptq import CharOffsetTokenizer

    prompt = (
        "任务文本 "
        "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|> "
        "中间说明 "
        "<|sid_begin|><s_a_4><s_b_5><s_c_6><|sid_end|> "
        "<|sid_begin|>"
    )

    groups = build_prompt_token_group_batches(
        tokenizer=CharOffsetTokenizer(),
        prompts=[prompt],
        device=torch.device("cpu"),
    )[0]

    assert groups.shape == (1, len(prompt))
    assert groups[0, prompt.index("任")].item() == PROMPT_TOKEN_GROUP_IDS["text"]
    assert groups[0, prompt.index("<s_a_1>") + 1].item() == PROMPT_TOKEN_GROUP_IDS["history_sid"]
    assert groups[0, prompt.index("<s_a_4>") + 1].item() == PROMPT_TOKEN_GROUP_IDS["interest_sid"]
    assert groups[0, prompt.index("<|sid_begin|>") + 1].item() == PROMPT_TOKEN_GROUP_IDS["sid_boundary"]
    assert groups[0, prompt.rindex("<|sid_begin|>") + 1].item() == PROMPT_TOKEN_GROUP_IDS["sid_boundary"]


def test_parse_args_accepts_grad_weighted_gptq_mode(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["prog", "--mode", "grad_weighted_gptq_fp8_w8a8"])

    args = parse_args()

    assert args.mode == "grad_weighted_gptq_fp8_w8a8"
    assert args.grad_weight_clip_percentile == 99.0
    assert args.grad_weight_floor == 0.05
    assert args.grad_weight_normalize_mean is True
