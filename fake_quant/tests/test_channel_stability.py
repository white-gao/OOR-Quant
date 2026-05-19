from __future__ import annotations

import torch
import pytest

from fake_quant.ranking_margin.channel_stability import (
    canonical_shared_input_name,
    topk_overlap_stats,
    summarize_channel_stability,
    topk_frequency,
    topk_jaccard_matrix,
)


def test_topk_jaccard_matrix_uses_sample_top_channels() -> None:
    scores = torch.tensor(
        [
            [10.0, 9.0, 1.0, 0.0],
            [8.0, 1.0, 7.0, 0.0],
            [0.0, 1.0, 9.0, 8.0],
        ]
    )

    jaccard = topk_jaccard_matrix(scores, topk_fraction=0.5)

    expected = torch.tensor(
        [
            [1.0, 1.0 / 3.0, 0.0],
            [1.0 / 3.0, 1.0, 1.0 / 3.0],
            [0.0, 1.0 / 3.0, 1.0],
        ]
    )
    torch.testing.assert_close(jaccard, expected)


def test_topk_frequency_counts_how_often_each_channel_is_selected() -> None:
    scores = torch.tensor(
        [
            [10.0, 9.0, 1.0, 0.0],
            [8.0, 1.0, 7.0, 0.0],
            [0.0, 1.0, 9.0, 8.0],
        ]
    )

    frequency = topk_frequency(scores, topk_fraction=0.5)

    torch.testing.assert_close(frequency, torch.tensor([2.0, 1.0, 2.0, 1.0]))


def test_summarize_channel_stability_reports_jaccard_and_cv() -> None:
    scores = torch.tensor(
        [
            [1.0, 2.0, 4.0],
            [1.0, 2.0, 4.0],
            [1.0, 2.0, 4.0],
        ]
    )

    summary = summarize_channel_stability(scores, topk_fractions=[1.0 / 3.0])

    assert summary["num_samples"] == 3
    assert summary["num_channels"] == 3
    assert summary["topk/0.333333/jaccard_mean"] == 1.0
    assert summary["topk/0.333333/frequency_max"] == 3.0
    assert summary["cv_mean"] == 0.0


def test_canonical_shared_input_name_keeps_only_requested_nodes() -> None:
    assert (
        canonical_shared_input_name("model.layers.7.self_attn.q_proj")
        == "model.layers.7.attn_qkv_input"
    )
    assert (
        canonical_shared_input_name("model.layers.7.mlp.gate_proj")
        == "model.layers.7.ffn_gate_up_input"
    )
    assert canonical_shared_input_name("model.layers.7.self_attn.o_proj") is None


def test_topk_overlap_stats_compares_two_channel_rankings_per_sample() -> None:
    ranking_scores = torch.tensor(
        [
            [10.0, 9.0, 1.0, 0.0],
            [8.0, 1.0, 7.0, 0.0],
        ]
    )
    activation_scores = torch.tensor(
        [
            [10.0, 1.0, 9.0, 0.0],
            [0.0, 8.0, 7.0, 1.0],
        ]
    )

    stats = topk_overlap_stats(ranking_scores, activation_scores, topk_fraction=0.5)

    assert stats["k"] == 2
    assert stats["jaccard_mean"] == pytest.approx(1.0 / 3.0)
    assert stats["overlap_count_mean"] == pytest.approx(1.0)
