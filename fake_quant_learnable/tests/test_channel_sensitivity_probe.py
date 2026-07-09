from __future__ import annotations

import torch
import pytest

from fake_quant_learnable.probe_channel_sensitivity import (
    channel_profile_stats,
    channel_topk_overlap,
    cosine_similarity,
    token_profile_consistency_stats,
)


def test_channel_profile_stats_reports_concentration() -> None:
    profile = torch.tensor([9.0, 1.0, 0.0, 0.0])

    stats = channel_profile_stats(profile, topk=(1, 2))

    assert stats["num_channels"] == 4
    assert stats["top1_share"] == 0.9
    assert stats["top2_share"] == 1.0
    assert stats["max_to_mean"] == 3.6


def test_channel_profile_similarity_and_overlap() -> None:
    sid_a = torch.tensor([4.0, 3.0, 1.0, 0.0])
    sid_b = torch.tensor([0.0, 1.0, 3.0, 4.0])

    assert cosine_similarity(sid_a, sid_b) < 0.5
    assert channel_topk_overlap(sid_a, sid_b, k=2) == 0.0
    assert channel_topk_overlap(sid_a, sid_b, k=3) == pytest.approx(2.0 / 3.0)


def test_token_profile_consistency_stats_separates_intra_from_inter_profiles() -> None:
    sid_a_profiles = [
        torch.tensor([4.0, 3.0, 0.0, 0.0]),
        torch.tensor([3.9, 3.1, 0.0, 0.0]),
    ]
    other_profiles = [
        torch.tensor([0.0, 0.0, 4.0, 3.0]),
        torch.tensor([0.0, 0.0, 3.8, 3.2]),
    ]

    stats = token_profile_consistency_stats(
        sid_a_profiles,
        other_profiles,
        topk=(2,),
        max_pairs=16,
    )

    assert stats["token_profiles"] == 2
    assert stats["intra_pairs"] == 1
    assert stats["inter_pairs"] == 4
    assert stats["intra_cosine"] > 0.99
    assert stats["avg_inter_cosine"] == 0.0
    assert stats["cosine_separation"] > 0.99
    assert stats["intra_top2_overlap"] == 1.0
    assert stats["avg_inter_top2_overlap"] == 0.0
