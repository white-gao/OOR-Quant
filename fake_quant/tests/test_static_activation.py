from __future__ import annotations

import torch
import torch.nn as nn

from fake_quant.apply import _shared_prepare_input, apply_fp8_fake_quant
from fake_quant.modules import FakeQuantLinear
from fake_quant.smoothquant.core import save_activation_absmax
from fake_quant.static_activation import (
    compute_static_tensor_activation_scales_for_model,
    static_tensor_scale_from_absmax,
)
from fake_quant.quant import FP8_MAX


def test_static_tensor_scale_from_absmax_uses_layer_max() -> None:
    x_absmax = torch.tensor([1.0, 2.0, 4.0])

    scale = static_tensor_scale_from_absmax(x_absmax)

    torch.testing.assert_close(scale, torch.tensor(4.0 / FP8_MAX))


def test_static_tensor_scales_follow_quantized_linear_filters() -> None:
    model = nn.Sequential(
        nn.Linear(3, 2),
        nn.Sequential(nn.Linear(3, 2)),
    )
    activation_absmax = {
        "0": torch.tensor([1.0, 2.0, 4.0]),
        "1.0": torch.tensor([8.0, 2.0, 1.0]),
    }

    scales = compute_static_tensor_activation_scales_for_model(
        model,
        activation_absmax,
        target_regex=r"^1\.",
    )

    assert set(scales) == {"1.0"}
    torch.testing.assert_close(scales["1.0"], torch.tensor(8.0 / FP8_MAX))


def test_shared_prepare_input_uses_stored_static_scales_without_linear_requant() -> None:
    linear_a = FakeQuantLinear(
        nn.Linear(3, 2),
        act_quant="none",
        activation_static_scale=torch.tensor(2.0 / FP8_MAX),
    )
    linear_b = FakeQuantLinear(
        nn.Linear(3, 2),
        act_quant="none",
        activation_static_scale=torch.tensor(8.0 / FP8_MAX),
    )
    x = torch.tensor([[[1.0, 2.0, 4.0]]])

    prepared = _shared_prepare_input((linear_a, linear_b), x)

    # Shared input uses the max static scale across consumers.
    expected_scale = torch.tensor(8.0 / FP8_MAX)
    expected = (torch.clamp(x / expected_scale, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * expected_scale)
    torch.testing.assert_close(prepared, expected)


def test_shared_input_static_activation_keeps_scales_on_wrapped_linears(tmp_path) -> None:
    model = nn.Sequential(nn.Linear(3, 2))
    scales_path = tmp_path / "x_absmax.pt"
    save_activation_absmax(
        scales_path,
        activation_absmax={"0": torch.tensor([1.0, 2.0, 8.0])},
        metadata={},
    )

    apply_fp8_fake_quant(
        model,
        act_quant="static_tensor",
        act_quant_mode="shared_input",
        activation_absmax_path=str(scales_path),
    )

    assert isinstance(model[0], FakeQuantLinear)
    assert model[0].act_quant == "none"
    torch.testing.assert_close(model[0].activation_static_scale.cpu(), torch.tensor(8.0 / FP8_MAX))
