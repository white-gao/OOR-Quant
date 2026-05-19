from __future__ import annotations

import torch

from fake_quant.ranking_margin.core import (
    compute_ranking_margin_smooth_scales_for_model,
    compute_ranking_margin_smooth_scale,
    normalize_importance,
)
from fake_quant.smoothquant.core import compute_smooth_scale


def test_importance_normalization_uses_geometric_mean_and_clip() -> None:
    importance = torch.tensor([16.0, 1.0, 1.0 / 16.0])

    normalized = normalize_importance(importance, clip_min=0.5, clip_max=2.0)

    expected = torch.tensor([2.0, 1.0, 0.5])
    torch.testing.assert_close(normalized, expected)


def test_beta_zero_recovers_plain_smoothquant_scale() -> None:
    activation_absmax = torch.tensor([16.0, 4.0, 1.0])
    weight_absmax = torch.tensor([1.0, 4.0, 16.0])
    importance = torch.tensor([16.0, 1.0, 1.0 / 16.0])

    plain = compute_smooth_scale(activation_absmax, weight_absmax, alpha=0.5)
    ranked = compute_ranking_margin_smooth_scale(
        activation_absmax,
        weight_absmax,
        importance,
        alpha=0.5,
        beta=0.0,
    )

    torch.testing.assert_close(ranked, plain)


def test_ranking_importance_increases_scale_for_sensitive_channels() -> None:
    activation_absmax = torch.tensor([16.0, 4.0, 1.0])
    weight_absmax = torch.tensor([1.0, 4.0, 16.0])
    importance = torch.tensor([16.0, 1.0, 1.0 / 16.0])

    scale = compute_ranking_margin_smooth_scale(
        activation_absmax,
        weight_absmax,
        importance,
        alpha=0.5,
        beta=0.25,
        clip_min=1.0 / 16.0,
        clip_max=16.0,
    )

    # Plain SmoothQuant gives [4, 1, 0.25]. Importance correction multiplies
    # by [2, 1, 0.5], protecting the first channel and relaxing the third.
    expected = torch.tensor([8.0, 1.0, 0.125])
    torch.testing.assert_close(scale, expected)


class _TinyAttentionBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = torch.nn.Linear(3, 2, bias=False)
        torch.nn.init.ones_(self.self_attn.q_proj.weight)


class _TinyLayerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_TinyAttentionBlock(), _TinyAttentionBlock()])


def test_ranking_layer_cutoff_uses_plain_smoothquant_for_higher_layers() -> None:
    model = _TinyLayerModel()
    activation_absmax = {
        "model.layers.0.self_attn.q_proj": torch.tensor([16.0, 4.0, 1.0]),
        "model.layers.1.self_attn.q_proj": torch.tensor([16.0, 4.0, 1.0]),
    }
    rank_importance = {
        "model.layers.0.self_attn.q_proj": torch.tensor([16.0, 1.0, 1.0 / 16.0]),
        "model.layers.1.self_attn.q_proj": torch.tensor([16.0, 1.0, 1.0 / 16.0]),
    }

    scales = compute_ranking_margin_smooth_scales_for_model(
        model,
        activation_absmax,
        rank_importance,
        alpha=0.5,
        beta=0.25,
        ranking_layer_cutoff=1,
    )

    plain = compute_smooth_scale(
        activation_absmax["model.layers.1.self_attn.q_proj"],
        torch.tensor([1.0, 1.0, 1.0]),
        alpha=0.5,
    )
    ranked = compute_ranking_margin_smooth_scale(
        activation_absmax["model.layers.0.self_attn.q_proj"],
        torch.tensor([1.0, 1.0, 1.0]),
        rank_importance["model.layers.0.self_attn.q_proj"],
        alpha=0.5,
        beta=0.25,
    )

    torch.testing.assert_close(scales["model.layers.0.self_attn.q_proj"], ranked)
    torch.testing.assert_close(scales["model.layers.1.self_attn.q_proj"], plain)


def test_ranking_layer_min_uses_plain_smoothquant_for_lower_layers() -> None:
    model = _TinyLayerModel()
    activation_absmax = {
        "model.layers.0.self_attn.q_proj": torch.tensor([16.0, 4.0, 1.0]),
        "model.layers.1.self_attn.q_proj": torch.tensor([16.0, 4.0, 1.0]),
    }
    rank_importance = {
        "model.layers.0.self_attn.q_proj": torch.tensor([16.0, 1.0, 1.0 / 16.0]),
        "model.layers.1.self_attn.q_proj": torch.tensor([16.0, 1.0, 1.0 / 16.0]),
    }

    scales = compute_ranking_margin_smooth_scales_for_model(
        model,
        activation_absmax,
        rank_importance,
        alpha=0.5,
        beta=0.25,
        ranking_layer_min=1,
    )

    plain = compute_smooth_scale(
        activation_absmax["model.layers.0.self_attn.q_proj"],
        torch.tensor([1.0, 1.0, 1.0]),
        alpha=0.5,
    )
    ranked = compute_ranking_margin_smooth_scale(
        activation_absmax["model.layers.1.self_attn.q_proj"],
        torch.tensor([1.0, 1.0, 1.0]),
        rank_importance["model.layers.1.self_attn.q_proj"],
        alpha=0.5,
        beta=0.25,
    )

    torch.testing.assert_close(scales["model.layers.0.self_attn.q_proj"], plain)
    torch.testing.assert_close(scales["model.layers.1.self_attn.q_proj"], ranked)
