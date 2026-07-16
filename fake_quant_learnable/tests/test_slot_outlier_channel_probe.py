from __future__ import annotations

import pytest
import torch

from fake_quant_learnable.probe_slot_outlier_channels import (
    TokenwiseOutlierAccumulator,
    TokenwiseOutlierProbeResult,
    average_inter_token_overlap,
    average_intra_token_overlap,
    build_probe_outputs,
    collect_slot_outlier_channels,
    tokenwise_topk_mask,
)


def test_tokenwise_topk_mask_selects_channels_for_each_token_before_aggregation() -> None:
    activations = torch.tensor(
        [
            [9.0, 8.0, 1.0, 0.0],
            [0.0, 1.0, 8.0, 9.0],
        ]
    )

    mask = tokenwise_topk_mask(activations, fraction=0.5)

    assert mask.dtype == torch.bool
    assert mask.sum(dim=1).tolist() == [2, 2]
    assert mask[0].tolist() == [True, True, False, False]
    assert mask[1].tolist() == [False, False, True, True]


def test_tokenwise_accumulator_counts_topk_occurrences_per_channel_and_split() -> None:
    accumulator = TokenwiseOutlierAccumulator(outlier_fraction=0.5)
    accumulator.add(
        torch.tensor([[9.0, 8.0, 1.0, 0.0], [0.0, 1.0, 8.0, 9.0]]),
        split_id=0,
    )
    accumulator.add(torch.tensor([[7.0, 0.0, 6.0, 1.0]]), split_id=1)

    profile = accumulator.profile()

    torch.testing.assert_close(profile["channel_counts"], torch.tensor([2, 1, 2, 1]))
    torch.testing.assert_close(profile["split0_channel_counts"], torch.tensor([1, 1, 1, 1]))
    torch.testing.assert_close(profile["split1_channel_counts"], torch.tensor([1, 0, 1, 0]))
    torch.testing.assert_close(
        profile["channel_frequency"],
        torch.tensor([2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0]),
    )
    assert profile["token_count"] == 3
    assert profile["topk"] == 2


def test_exact_intra_and_inter_token_overlap_from_channel_counts() -> None:
    first_counts = torch.tensor([2, 1, 1, 0])
    second_counts = torch.tensor([0, 1, 1, 2])

    intra = average_intra_token_overlap(first_counts, token_count=2, topk=2)
    inter = average_inter_token_overlap(
        first_counts,
        token_count_a=2,
        counts_b=second_counts,
        token_count_b=2,
        topk=2,
    )

    assert intra == pytest.approx(0.5)
    assert inter == pytest.approx(0.25)


def test_collect_slot_outlier_channels_keeps_tokenwise_counts_and_representative_tokens() -> None:
    class TinyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4, bias=False)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.q_proj(hidden_states)

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([TinyLayer()])

        def forward(self, input_ids: torch.Tensor, **_kwargs):
            hidden = input_ids.float()
            for layer in self.layers:
                hidden = layer(hidden)
            return hidden

    model = TinyModel()
    first = torch.tensor([[[9.0, 8.0, 1.0, 0.0], [0.0, 1.0, 8.0, 9.0]]])
    second = torch.tensor([[[7.0, 0.0, 6.0, 1.0], [1.0, 9.0, 0.0, 8.0]]])
    groups = torch.tensor([[0, 1]])

    result = collect_slot_outlier_channels(
        model=model,
        layers=model.layers,
        layer_indices=[0],
        plot_layer_indices=[0],
        model_batches=[
            {"input_ids": first, "attention_mask": torch.ones(1, 2)},
            {"input_ids": second, "attention_mask": torch.ones(1, 2)},
        ],
        slot_group_batches=[groups, groups],
        split_ids=[0, 1],
        representative_index=1,
        outlier_fraction=0.5,
        linear_regex=r"q_proj$",
        progress_every=0,
    )

    text = result.accumulators[(0, "q_proj", "text")].profile()
    torch.testing.assert_close(text["channel_counts"], torch.tensor([2, 1, 1, 0]))
    captured = result.representative[(0, "q_proj")]
    torch.testing.assert_close(captured.activations, second[0])
    torch.testing.assert_close(captured.groups, groups[0])


def test_build_probe_outputs_reports_tokenwise_intra_and_inter_group_overlap() -> None:
    result = TokenwiseOutlierProbeResult(outlier_fraction=0.5)
    values = {
        "text": torch.tensor([[9.0, 8.0, 1.0, 0.0], [8.0, 1.0, 7.0, 0.0]]),
        "sid_a": torch.tensor([[0.0, 1.0, 8.0, 9.0], [1.0, 0.0, 9.0, 8.0]]),
        "sid_b": torch.tensor([[8.0, 7.0, 0.0, 1.0], [7.0, 8.0, 1.0, 0.0]]),
        "sid_c": torch.tensor([[1.0, 0.0, 9.0, 8.0], [0.0, 1.0, 8.0, 9.0]]),
    }
    for group, activation in values.items():
        result.add(layer=0, module="q_proj", group=group, values=activation, split_id=0)
        result.add(layer=0, module="q_proj", group=group, values=activation, split_id=1)

    group_rows, pair_rows, layer_rows, _profiles = build_probe_outputs(result)

    text_row = next(row for row in group_rows if row["group"] == "text")
    assert text_row["intra_token_overlap"] == pytest.approx(2.0 / 3.0)
    text_sid_a = next(
        row for row in pair_rows if row["group_a"] == "text" and row["group_b"] == "sid_a"
    )
    assert text_sid_a["inter_token_overlap"] == pytest.approx(0.25)
    assert text_sid_a["pair_intra_minus_inter"] > 0.0
    summary = next(row for row in layer_rows if row["layer"] == 0 and row["module"] == "q_proj")
    assert summary["main_group_intra_token_overlap"] > summary["text_vs_sid_inter_token_overlap"]
