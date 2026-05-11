from __future__ import annotations

import torch
import torch.nn.functional as F

from fake_quant.smoothquant.core import compute_smooth_scale, smooth_linear_weight


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
