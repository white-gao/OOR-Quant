from __future__ import annotations

import torch
import torch.nn as nn

from fake_quant_learnable.modules import BaselineFakeQuantLinear, FrozenLearnedFakeQuantLinear
from fake_quant_learnable.run_m1_onerec_ad import (
    apply_baseline_layers,
    apply_learned_quant_params_to_layers,
    calibrate_model_layers_m1,
    capture_layer_input_batches,
    get_transformer_layers,
    load_ad_data,
    parse_layer_indices,
)


class ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = ToyBackbone()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        del attention_mask
        return self.model(input_ids.float())


def test_calibrate_exports_params_that_can_be_loaded_into_fresh_model() -> None:
    torch.manual_seed(7)
    model = ToyModel()
    base_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    batches = [{"input_ids": torch.randn(3, 4), "attention_mask": torch.ones(3, 4)}]
    learned_quant_params = {}

    calibrate_model_layers_m1(
        model=model,
        model_batches=batches,
        layer_indices=[1],
        steps=2,
        lr=0.01,
        act_quant="per_token",
        init_clip_multiplier=0.5,
        enable_let=True,
        learned_quant_params=learned_quant_params,
    )

    fresh = ToyModel()
    fresh.load_state_dict(base_state)
    applied = apply_learned_quant_params_to_layers(
        fresh,
        {"format_version": 1, "method": "m2_lwt_let", "layers": learned_quant_params},
    )

    x = torch.randn(2, 4)
    assert applied == [1]
    assert isinstance(fresh.model.layers[1], FrozenLearnedFakeQuantLinear)
    torch.testing.assert_close(fresh.model.layers[1](x), model.model.layers[1](x), rtol=0, atol=0)


def test_parse_layer_indices_supports_all_last_and_csv() -> None:
    assert parse_layer_indices("all", num_layers=4) == [0, 1, 2, 3]
    assert parse_layer_indices("last:2", num_layers=4) == [2, 3]
    assert parse_layer_indices("0,2-3", num_layers=4) == [0, 2, 3]


def test_get_transformer_layers_reads_model_model_layers() -> None:
    model = ToyModel()

    layers = get_transformer_layers(model)

    assert layers is model.model.layers
    assert len(layers) == 3


def test_load_ad_data_applies_offset_after_loading_enough_samples(monkeypatch) -> None:
    created_loaders = []

    class DummyLoader:
        def __init__(self) -> None:
            self.requests = []

        def load_data(self, split: str, sample_size: int):
            self.requests.append((split, sample_size))
            return {f"sample_{idx}": {"prompt": str(idx)} for idx in range(sample_size)}

    def fake_get_loader(**_kwargs):
        loader = DummyLoader()
        created_loaders.append(loader)
        return loader

    monkeypatch.setattr("fake_quant_learnable.run_m1_onerec_ad.get_loader", fake_get_loader)

    data = load_ad_data(
        tokenizer=None,
        data_dir="data/onerec_data/benchmark-data",
        split="test",
        sample_size=3,
        sample_offset=5,
    )

    assert created_loaders[0].requests == [("test", 8)]
    assert list(data.keys()) == ["sample_5", "sample_6", "sample_7"]


def test_capture_layer_input_batches_records_args_and_kwargs() -> None:
    torch.manual_seed(0)
    model = ToyModel()
    batches = [
        {"input_ids": torch.randn(2, 4), "attention_mask": torch.ones(2, 4)},
        {"input_ids": torch.randn(2, 4), "attention_mask": torch.ones(2, 4)},
    ]

    captured = capture_layer_input_batches(
        model=model,
        layer=model.model.layers[1],
        model_batches=batches,
    )

    assert len(captured) == 2
    args, kwargs = captured[0]
    assert len(args) == 1
    assert args[0].shape == (2, 4)
    assert kwargs == {}
    assert not args[0].requires_grad


def test_apply_baseline_layers_replaces_selected_layers_only() -> None:
    model = ToyModel()

    summaries = apply_baseline_layers(
        model=model,
        layer_indices=[0, 2],
        act_quant="per_token",
    )

    assert sorted(summaries.keys()) == [0, 2]
    assert summaries[0].replaced_linears == 1
    assert summaries[2].replaced_linears == 1
    assert isinstance(model.model.layers[0], BaselineFakeQuantLinear)
    assert isinstance(model.model.layers[1], nn.Linear)
    assert isinstance(model.model.layers[2], BaselineFakeQuantLinear)


def test_calibrate_model_layers_m1_can_enable_let() -> None:
    torch.manual_seed(4)
    model = ToyModel()
    batches = [{"input_ids": torch.randn(3, 4), "attention_mask": torch.ones(3, 4)}]

    histories = calibrate_model_layers_m1(
        model=model,
        model_batches=batches,
        layer_indices=[1],
        steps=2,
        lr=0.01,
        act_quant="per_token",
        init_clip_multiplier=0.5,
        enable_let=True,
    )

    assert list(histories.keys()) == [1]
    assert isinstance(model.model.layers[1], FrozenLearnedFakeQuantLinear)
    assert model.model.layers[1].let_scale is not None


def test_calibrate_model_layers_m1_replaces_selected_layer() -> None:
    torch.manual_seed(0)
    model = ToyModel()
    batches = [{"input_ids": torch.randn(3, 4), "attention_mask": torch.ones(3, 4)}]
    original_type = type(model.model.layers[1])

    histories = calibrate_model_layers_m1(
        model=model,
        model_batches=batches,
        layer_indices=[1],
        steps=2,
        lr=0.01,
        act_quant="per_token",
        init_clip_multiplier=0.5,
    )

    assert list(histories.keys()) == [1]
    assert type(model.model.layers[0]) is original_type
    assert type(model.model.layers[1]) is not original_type
    assert isinstance(model.model.layers[1], FrozenLearnedFakeQuantLinear)
    assert histories[1].initial_loss >= 0
