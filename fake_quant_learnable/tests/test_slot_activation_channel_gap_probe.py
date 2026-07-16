from __future__ import annotations

import pytest
import torch

from fake_quant_learnable.probe_slot_activation_channel_gap import (
    ActivationChannelAccumulator,
    average_intra_sample_cosine,
    channel_top_fraction_overlap,
    collect_activation_channel_gap,
    cosine_similarity,
)


def test_activation_channel_accumulator_computes_group_profile() -> None:
    accumulator = ActivationChannelAccumulator()
    values = torch.tensor(
        [
            [1.0, -2.0, 0.0],
            [3.0, 0.0, -4.0],
        ]
    )

    accumulator.add(values)
    profile = accumulator.profile()

    torch.testing.assert_close(profile["energy"], torch.tensor([5.0, 2.0, 8.0]))
    torch.testing.assert_close(profile["mean_abs"], torch.tensor([2.0, 1.0, 2.0]))
    torch.testing.assert_close(profile["max_abs"], torch.tensor([3.0, 2.0, 4.0]))
    assert profile["token_count"] == 2
    assert profile["sample_count"] == 1


def test_channel_similarity_and_top_fraction_overlap_detect_slot_specific_channels() -> None:
    sid_a = torch.tensor([9.0, 8.0, 1.0, 0.0])
    sid_b = torch.tensor([0.0, 1.0, 8.0, 9.0])

    assert cosine_similarity(sid_a, sid_b) < 0.2
    assert channel_top_fraction_overlap(sid_a, sid_b, fraction=0.5) == 0.0
    assert channel_top_fraction_overlap(sid_a, sid_b, fraction=0.75) == pytest.approx(2.0 / 3.0)


def test_average_intra_sample_cosine_uses_streaming_normalized_profile_sum() -> None:
    profiles = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.9, 0.1, 0.0],
        ]
    )
    normalized = torch.nn.functional.normalize(profiles, dim=1)
    profile_sum = normalized.sum(dim=0)

    actual = average_intra_sample_cosine(profile_sum, sample_count=3)
    expected = sum(
        float(torch.dot(normalized[i], normalized[j]))
        for i in range(3)
        for j in range(i + 1, 3)
    ) / 3.0

    assert actual == pytest.approx(expected)


def test_collect_activation_channel_gap_applies_slot_masks_to_linear_inputs() -> None:
    class TinyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(2, 2, bias=False)

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
    inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
    groups = torch.tensor([[0, 1, 2, 3]])

    result = collect_activation_channel_gap(
        model=model,
        layers=model.layers,
        layer_indices=[0],
        model_batches=[{"input_ids": inputs, "attention_mask": torch.ones(1, 4)}],
        slot_group_batches=[groups],
        linear_regex=r"q_proj$",
        progress_every=0,
    )

    text = result.accumulators[(0, "q_proj", "text")].profile()
    sid_a = result.accumulators[(0, "q_proj", "sid_a")].profile()
    torch.testing.assert_close(text["energy"], torch.tensor([1.0, 4.0]))
    torch.testing.assert_close(sid_a["energy"], torch.tensor([9.0, 16.0]))
