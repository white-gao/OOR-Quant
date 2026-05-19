from __future__ import annotations

import torch
import torch.nn.functional as F

from fake_quant.apply import apply_smoothquant_fp8_fake_quant
from fake_quant.modules import FakeQuantLinear
from fake_quant.smoothquant.core import compute_smooth_scale, smooth_linear_weight
from fake_quant.smoothquant.core import SmoothQuantLinear


def test_compute_smooth_scale_uses_per_input_channel_stats() -> None:
    x_absmax = torch.tensor([16.0, 4.0, 1.0])
    w_absmax = torch.tensor([1.0, 4.0, 16.0])

    scale = compute_smooth_scale(x_absmax, w_absmax, alpha=0.5)

    expected = torch.tensor([4.0, 1.0, 0.25])
    torch.testing.assert_close(scale, expected)


def test_smooth_linear_transform_preserves_unquantized_linear_math() -> None:
    x = torch.tensor(
        [
            [[1.0, -2.0, 3.0]],
            [[-4.0, 5.0, -6.0]],
        ]
    )
    weight = torch.tensor(
        [
            [0.5, -1.0, 2.0],
            [-1.5, 0.25, 0.75],
        ]
    )
    bias = torch.tensor([0.1, -0.2])
    scale = torch.tensor([2.0, 0.5, 4.0])

    baseline = F.linear(x, weight, bias)
    smoothed = F.linear(x / scale, smooth_linear_weight(weight, scale), bias)

    torch.testing.assert_close(smoothed, baseline)


class _TinyAttentionBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = torch.nn.Linear(3, 2, bias=False)


class _TinyLayerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_TinyAttentionBlock(), _TinyAttentionBlock()])


def test_smooth_layer_cutoff_falls_back_to_plain_fp8_for_higher_layers(tmp_path) -> None:
    model = _TinyLayerModel()
    scale_path = tmp_path / "absmax.pt"
    torch.save(
        {
            "x_absmax": {
                "model.layers.0.self_attn.q_proj": torch.tensor([1.0, 2.0, 3.0]),
                "model.layers.1.self_attn.q_proj": torch.tensor([1.0, 2.0, 3.0]),
            }
        },
        scale_path,
    )

    apply_smoothquant_fp8_fake_quant(
        model,
        activation_absmax_path=str(scale_path),
        act_quant="none",
        smooth_layer_cutoff=1,
    )

    assert isinstance(model.model.layers[0].self_attn.q_proj, SmoothQuantLinear)
    assert isinstance(model.model.layers[1].self_attn.q_proj, FakeQuantLinear)


def test_smooth_layer_min_falls_back_to_plain_fp8_for_lower_layers(tmp_path) -> None:
    model = _TinyLayerModel()
    scale_path = tmp_path / "absmax.pt"
    torch.save(
        {
            "x_absmax": {
                "model.layers.0.self_attn.q_proj": torch.tensor([1.0, 2.0, 3.0]),
                "model.layers.1.self_attn.q_proj": torch.tensor([1.0, 2.0, 3.0]),
            }
        },
        scale_path,
    )

    apply_smoothquant_fp8_fake_quant(
        model,
        activation_absmax_path=str(scale_path),
        act_quant="none",
        smooth_layer_min=1,
    )

    assert isinstance(model.model.layers[0].self_attn.q_proj, FakeQuantLinear)
    assert isinstance(model.model.layers[1].self_attn.q_proj, SmoothQuantLinear)
