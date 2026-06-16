from __future__ import annotations

import torch
import torch.nn as nn

from fake_quant.smoothquant.core import compute_smooth_scale
from fake_quant_learnable.modules import BaselineFakeQuantLinear, FrozenLearnedFakeQuantLinear, LearnableFakeQuantLinear
from fake_quant_learnable.smoothquant_runtime import (
    apply_smoothquant_scales_to_learnable,
    collect_smoothquant_scales,
)
from fake_quant_learnable.run_m1_onerec_ad import (
    apply_baseline_layers,
    apply_learned_quant_params_to_layers,
    apply_smoothquant_layers,
    calibrate_model_layers_m1,
    capture_layer_input_batches,
    get_transformer_layers,
    load_ad_data,
    parse_args,
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


class TinyOmniAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class TinyOmniMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TinyOmniBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(4)
        self.self_attn = TinyOmniAttention()
        self.post_attention_layernorm = nn.LayerNorm(4)
        self.mlp = TinyOmniMLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.self_attn(self.input_layernorm(x))
        return self.mlp(self.post_attention_layernorm(x + attn))


class ToyOmniModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyOmniBlock()])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        del attention_mask
        x = input_ids.float()
        for layer in self.model.layers:
            x = layer(x)
        return x


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
        epochs=2,
        lwt_lr=0.01,
        let_lr=0.01,
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


def test_parse_args_attaches_compact_fixed_defaults(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["prog"])

    args = parse_args()

    assert args.mode == "m2_lwt_let"
    assert args.model_path == "/home/guowei/OneRec-1.7B/"
    assert args.data_dir == "data/onerec_data/benchmark-data-calib1024"
    assert args.act_quant == "per_token"
    assert args.act_quant_mode == "shared_input"
    assert args.smooth_scope == "omni"
    assert args.smooth_fold is True
    assert args.num_beams == 32
    assert args.num_return_sequences == 32
    assert args.max_new_tokens == 3


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


def test_collect_smoothquant_scales_uses_activation_and_weight_max() -> None:
    linear = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, 4.0], [2.0, 0.5]]))
    batches = [torch.tensor([[3.0, -8.0], [1.0, 2.0]])]

    scales = collect_smoothquant_scales(
        linear,
        batches,
        alpha=0.5,
        min_scale=0.0,
        max_scale=100.0,
    )

    expected = torch.sqrt(torch.tensor([3.0, 8.0]) / torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(scales[""], expected)


def test_collect_smoothquant_scales_matches_fake_quant_without_default_clamp() -> None:
    linear = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        linear.weight.fill_(1.0)
    batches = [torch.tensor([[10000.0]])]

    scales = collect_smoothquant_scales(linear, batches, alpha=0.5)

    expected = compute_smooth_scale(torch.tensor([10000.0]), torch.tensor([1.0]), alpha=0.5)
    torch.testing.assert_close(scales[""], expected)
    assert scales[""].item() > 20.0


def test_collect_smoothquant_scales_default_omni_scope_includes_down_proj() -> None:
    torch.manual_seed(10)
    block = TinyOmniBlock().eval()

    scales = collect_smoothquant_scales(block, [torch.randn(3, 4)], alpha=0.5)

    assert set(scales) == {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }
    torch.testing.assert_close(scales["self_attn.q_proj"], scales["self_attn.k_proj"])
    torch.testing.assert_close(scales["self_attn.q_proj"], scales["self_attn.v_proj"])
    torch.testing.assert_close(scales["mlp.gate_proj"], scales["mlp.up_proj"])


def test_apply_smoothquant_layers_folds_default_omni_scope_linears() -> None:
    torch.manual_seed(12)
    model = ToyOmniModel()
    batches = [{"input_ids": torch.randn(3, 4), "attention_mask": torch.ones(3, 4)}]

    summaries = apply_smoothquant_layers(
        model=model,
        model_batches=batches,
        layer_indices=[0],
        act_quant="per_token",
        smoothquant_alpha=0.5,
    )

    layer = model.model.layers[0]
    assert summaries[0].replaced_linears == 7
    assert isinstance(layer.self_attn.q_proj, FrozenLearnedFakeQuantLinear)
    assert isinstance(layer.self_attn.o_proj, FrozenLearnedFakeQuantLinear)
    assert isinstance(layer.mlp.gate_proj, FrozenLearnedFakeQuantLinear)
    assert isinstance(layer.mlp.up_proj, FrozenLearnedFakeQuantLinear)
    assert isinstance(layer.mlp.down_proj, FrozenLearnedFakeQuantLinear)
    assert layer.self_attn.q_proj.let_scale is None
    assert layer.self_attn.k_proj.let_scale is None
    assert layer.self_attn.v_proj.let_scale is None
    assert layer.mlp.gate_proj.let_scale is None
    assert layer.mlp.up_proj.let_scale is None
    assert layer.mlp.down_proj.let_scale is None


def test_apply_smoothquant_layers_can_disable_fold() -> None:
    torch.manual_seed(13)
    model = ToyOmniModel()
    batches = [{"input_ids": torch.randn(3, 4), "attention_mask": torch.ones(3, 4)}]

    apply_smoothquant_layers(
        model=model,
        model_batches=batches,
        layer_indices=[0],
        act_quant="per_token",
        smoothquant_alpha=0.5,
        smooth_fold=False,
    )

    layer = model.model.layers[0]
    assert isinstance(layer.self_attn.q_proj, FrozenLearnedFakeQuantLinear)
    assert layer.self_attn.q_proj.let_scale is not None
    assert layer.mlp.gate_proj.let_scale is not None
    assert isinstance(layer.mlp.down_proj, FrozenLearnedFakeQuantLinear)
    assert layer.mlp.down_proj.let_scale is not None


def test_smoothquant_let_initialization_is_clamped_to_learnable_bounds() -> None:
    target = LearnableFakeQuantLinear(nn.Linear(1, 1, bias=False), enable_let=True)

    applied = apply_smoothquant_scales_to_learnable(target, {"": torch.tensor([100.0])})

    assert applied == 1
    torch.testing.assert_close(target.let_scale, torch.tensor([target.max_let_scale]))


def test_apply_smoothquant_layers_replaces_selected_layer() -> None:
    torch.manual_seed(9)
    model = ToyModel()
    batches = [{"input_ids": torch.randn(3, 4), "attention_mask": torch.ones(3, 4)}]

    summaries = apply_smoothquant_layers(
        model=model,
        model_batches=batches,
        layer_indices=[1],
        act_quant="per_token",
        smoothquant_alpha=0.5,
        smoothquant_min_scale=0.05,
        smoothquant_max_scale=20.0,
    )

    assert list(summaries.keys()) == [1]
    assert summaries[1].replaced_linears == 1
    assert isinstance(model.model.layers[1], FrozenLearnedFakeQuantLinear)
    assert model.model.layers[1].let_scale is not None
    assert model.model.layers[1](torch.randn(2, 4)).shape == (2, 4)


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
        epochs=2,
        lwt_lr=0.01,
        let_lr=0.01,
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
        epochs=2,
        lwt_lr=0.01,
        act_quant="per_token",
        init_clip_multiplier=0.5,
    )

    assert list(histories.keys()) == [1]
    assert type(model.model.layers[0]) is original_type
    assert type(model.model.layers[1]) is not original_type
    assert isinstance(model.model.layers[1], FrozenLearnedFakeQuantLinear)
    assert histories[1].initial_loss >= 0
