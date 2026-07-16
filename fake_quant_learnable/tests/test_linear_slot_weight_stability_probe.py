from __future__ import annotations

import pytest
import torch

from fake_quant_learnable.probe_linear_slot_weight_stability import (
    compare_group_profiles,
    normalize_group_energies,
)
from fake_quant_learnable.token_weights import SLOT_TOKEN_GROUPS


def test_normalize_group_energies_preserves_group_order() -> None:
    profile = normalize_group_energies(
        {
            "text": 2.0,
            "sid_a": 1.0,
            "sid_b": 1.0,
            "sid_c": 0.0,
            "boundary": 0.0,
        }
    )

    assert tuple(profile.shape) == (len(SLOT_TOKEN_GROUPS),)
    torch.testing.assert_close(profile, torch.tensor([0.5, 0.25, 0.25, 0.0, 0.0]))


def test_compare_group_profiles_is_perfect_for_identical_profiles() -> None:
    profile = torch.tensor([0.5, 0.3, 0.1, 0.05, 0.05])

    stats = compare_group_profiles(profile, profile)

    assert stats["cosine"] == pytest.approx(1.0)
    assert stats["l1"] == pytest.approx(0.0)
    assert stats["spearman"] == pytest.approx(1.0)
    assert stats["top1_agree"] is True
    assert stats["top2_overlap"] == pytest.approx(1.0)


def test_compare_group_profiles_detects_reversed_group_priority() -> None:
    first = torch.tensor([0.6, 0.25, 0.1, 0.03, 0.02])
    second = torch.tensor([0.02, 0.03, 0.1, 0.25, 0.6])

    stats = compare_group_profiles(first, second)

    assert stats["cosine"] < 0.3
    assert stats["l1"] > 1.0
    assert stats["spearman"] < -0.9
    assert stats["top1_agree"] is False
    assert stats["top2_overlap"] == pytest.approx(0.0)
