from __future__ import annotations

import torch
import fake_quant_learnable.run_m1_onerec_ad as runner
import torch.nn as nn

from fake_quant_learnable.apply import apply_baseline_w8a8
from fake_quant_learnable.gptq import (
    _stable_cholesky_inverse_factor,
    gptaq_fp8_quantize_weight,
    gptq_fp8_quantize_weight,
)
from fake_quant_learnable.modules import (
    BaselineFakeQuantLinear,
    GPTQFakeQuantLinear,
    SmoothQuantFakeQuantLinear,
)
from fake_quant_learnable.quant import FP8_MAX, fp8_e4m3_qdq_forward, fp8_weight_per_channel_forward
from fake_quant_learnable.support.smoothquant_core import compute_smooth_scale
from fake_quant_learnable.support.smoothquant_runtime import collect_smoothquant_scales
from fake_quant_learnable.run_m1_onerec_ad import (
    apply_baseline_layers,
    apply_gptq_fp8_layers,
    apply_smoothquant_layers,
    capture_layer_input_batches,
    default_calib_split,
    get_transformer_layers,
    load_ad_data,
    maybe_evaluate,
    parse_args,
    parse_layer_indices,
    result_path,
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


def test_fp8_e4m3_qdq_forward_matches_torch_float8_cast() -> None:
    assert hasattr(torch, "float8_e4m3fn")
    x = torch.tensor(
        [
            [-1.37, -0.52, 0.0, 0.31],
            [0.78, 1.91, 4.40, -7.10],
        ],
        dtype=torch.float32,
    )
    scale = torch.tensor([[0.03], [0.17]], dtype=torch.float32)

    expected = (
        torch.clamp(x / scale, min=-FP8_MAX, max=FP8_MAX).to(torch.float8_e4m3fn).float()
        * scale
    )

    actual = fp8_e4m3_qdq_forward(x, scale)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_apply_baseline_w8a8_replaces_linears_skips_lm_head_and_has_no_parameters() -> None:
    class TinyModelWithHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
            self.lm_head = nn.Linear(4, 4)

    model = TinyModelWithHead()

    summary = apply_baseline_w8a8(model, act_quant="per_token")

    assert summary.replaced_linears == 2
    assert summary.skipped_linears == 1
    assert isinstance(model.backbone[0], BaselineFakeQuantLinear)
    assert isinstance(model.backbone[2], BaselineFakeQuantLinear)
    assert isinstance(model.lm_head, nn.Linear)
    for module in model.modules():
        if isinstance(module, BaselineFakeQuantLinear):
            assert list(module.parameters()) == []


def test_parse_layer_indices_supports_all_last_and_csv() -> None:
    assert parse_layer_indices("all", num_layers=4) == [0, 1, 2, 3]
    assert parse_layer_indices("last:2", num_layers=4) == [2, 3]
    assert parse_layer_indices("0,2-3", num_layers=4) == [0, 2, 3]


def test_parse_args_attaches_compact_fixed_defaults(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["prog"])

    args = parse_args()

    assert args.task == "ad"
    assert args.mode == "baseline_w8a8"
    assert args.model_path == "/home/guowei/OneRec-1.7B/"
    assert args.data_dir == "data/onerec_data/benchmark-data-calib1024"
    assert args.act_quant == "per_token"
    assert args.act_quant_mode == "shared_input"
    assert args.smooth_scope == "omni"
    assert args.smooth_fold is True
    assert args.num_beams == 32
    assert args.num_return_sequences == 32
    assert args.max_new_tokens == 3
    assert not hasattr(args, "epochs")
    assert not hasattr(args, "lwt_lr")
    assert not hasattr(args, "let_lr")


def test_parse_args_accepts_gptq_fp8_w8a8_mode(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["prog", "--mode", "gptq_fp8_w8a8"])

    args = parse_args()

    assert args.mode == "gptq_fp8_w8a8"
    assert args.gptq_damp_percent == 0.01
    assert args.gptq_block_size == 128


def test_parse_args_accepts_full_precision_mode(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["prog", "--mode", "full_precision"])

    args = parse_args()

    assert args.mode == "full_precision"


def test_parse_args_accepts_product_task_and_grad_tail1_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--task", "product", "--mode", "grad_weighted_gptq_fp8_w8a8_tail1"],
    )

    args = parse_args()

    assert args.task == "product"
    assert args.mode == "grad_weighted_gptq_fp8_w8a8_tail1"


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


def test_load_task_data_passes_requested_task_to_loader(monkeypatch) -> None:
    created_loader_kwargs = []

    class DummyLoader:
        def load_data(self, split: str, sample_size: int):
            return {f"sample_{idx}": {"prompt": str(idx)} for idx in range(sample_size)}

    def fake_get_loader(**kwargs):
        created_loader_kwargs.append(kwargs)
        return DummyLoader()

    monkeypatch.setattr("fake_quant_learnable.run_m1_onerec_ad.get_loader", fake_get_loader)

    data = runner.load_task_data(
        task_name="video",
        tokenizer=None,
        data_dir="data/onerec_data/benchmark-data-calib1024",
        split="calib",
        sample_size=2,
    )

    assert created_loader_kwargs[0]["task_name"] == "video"
    assert list(data.keys()) == ["sample_0", "sample_1"]


def test_default_calib_split_is_task_specific(tmp_path) -> None:
    data_dir = tmp_path / "benchmark-data-calib1024"
    (data_dir / "product").mkdir(parents=True)
    (data_dir / "product" / "product_calib.parquet").write_bytes(b"placeholder")

    assert default_calib_split(data_dir, "test", task_name="product") == "calib"
    assert default_calib_split(data_dir, "test", task_name="video") == "test"


def test_result_path_uses_task_name() -> None:
    path = result_path("fake_quant_learnable/results/example", "OneRec-1.7B", "product", "test")

    assert str(path).endswith("fake_quant_learnable/results/example/OneRec-1.7B/product/test_generated.json")


def test_maybe_evaluate_passes_requested_task(monkeypatch) -> None:
    calls = []

    def fake_evaluate_dev(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("fake_quant_learnable.run_m1_onerec_ad.Benchmark.evaluate_dev", fake_evaluate_dev)

    maybe_evaluate(
        output_dir="fake_quant_learnable/results/example",
        data_dir="data/onerec_data/benchmark-data-calib1024",
        overwrite=True,
        task_name="video",
    )

    assert calls[0]["task_types"] == ["video"]


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


def test_apply_baseline_layers_replaces_selected_layer_only() -> None:
    model = ToyModel()

    summaries = apply_baseline_layers(
        model=model,
        layer_indices=[1],
        act_quant="per_token",
        act_quant_mode="shared_input",
    )

    assert set(summaries) == {1}
    assert isinstance(model.model.layers[1], BaselineFakeQuantLinear)
    assert isinstance(model.model.layers[0], nn.Linear)
    assert isinstance(model.model.layers[2], nn.Linear)


def test_stable_cholesky_inverse_factor_handles_small_negative_numerical_eigenvalue() -> None:
    hessian = torch.diag(torch.tensor([1.0, 0.75, -1e-4], dtype=torch.float32))

    factor = _stable_cholesky_inverse_factor(hessian, eps=1e-12)

    assert factor.shape == hessian.shape
    assert torch.isfinite(factor).all()


def test_gptq_fp8_quantize_weight_reduces_calibrated_reconstruction_error() -> None:
    torch.manual_seed(21)
    weight = torch.tensor(
        [
            [1.000, 1.031, -0.812, 0.457],
            [-0.742, -0.701, 0.531, -0.266],
            [0.125, -0.094, 0.875, 0.344],
        ],
        dtype=torch.float32,
    )
    x = torch.tensor(
        [
            [1.0, 0.97, 0.05, -0.02],
            [0.91, 1.02, -0.04, 0.01],
            [-0.82, -0.88, 0.03, 0.02],
            [0.04, -0.02, 1.00, 0.92],
            [-0.03, 0.05, -0.91, -1.00],
        ],
        dtype=torch.float32,
    )
    hessian = x.t().matmul(x) / x.shape[0]

    naive_weight = fp8_weight_per_channel_forward(weight)
    gptq_weight = gptq_fp8_quantize_weight(
        weight,
        hessian,
        damp_percent=0.01,
        block_size=2,
    )

    target = x.matmul(weight.t())
    naive_mse = torch.mean((target - x.matmul(naive_weight.t())) ** 2)
    gptq_mse = torch.mean((target - x.matmul(gptq_weight.t())) ** 2)

    assert gptq_mse <= naive_mse


def test_gptaq_fp8_quantize_weight_matches_gptq_when_asymmetric_term_is_zero() -> None:
    torch.manual_seed(22)
    weight = torch.randn(5, 7, dtype=torch.float32)
    x = torch.randn(11, 7, dtype=torch.float32)
    hessian = x.t().matmul(x) / float(x.shape[0])
    dxx_t = torch.zeros_like(hessian)

    gptq_weight = gptq_fp8_quantize_weight(
        weight,
        hessian,
        damp_percent=0.01,
        block_size=3,
    )
    gptaq_weight = gptaq_fp8_quantize_weight(
        weight,
        hessian,
        dxx_t,
        alpha=0.25,
        damp_percent=0.01,
        block_size=3,
    )

    torch.testing.assert_close(gptaq_weight.float(), gptq_weight.float(), rtol=0, atol=0)


def test_apply_gptq_fp8_layers_replaces_selected_layer_with_gptq_wrappers() -> None:
    torch.manual_seed(23)
    model = ToyModel().eval()
    batches = [{"input_ids": torch.randn(2, 4), "attention_mask": torch.ones(2, 4)}]

    summaries = apply_gptq_fp8_layers(
        model=model,
        model_batches=batches,
        layer_indices=[1],
        act_quant="per_token",
        act_quant_mode="shared_input",
        damp_percent=0.01,
        block_size=2,
    )

    assert set(summaries) == {1}
    assert summaries[1].replaced_linears == 1
    assert isinstance(model.model.layers[1], GPTQFakeQuantLinear)
    assert isinstance(model.model.layers[0], nn.Linear)
    assert isinstance(model.model.layers[2], nn.Linear)
    assert list(model.model.layers[1].parameters()) == []


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
    torch.testing.assert_close(scales["mlp.gate_proj"], scales["mlp.up_proj"])


def test_apply_smoothquant_layers_uses_fixed_sq_wrappers_and_folds_known_paths() -> None:
    torch.manual_seed(13)
    model = ToyOmniModel().eval()
    batches = [{"input_ids": torch.randn(2, 4), "attention_mask": torch.ones(2, 4)}]

    summaries = apply_smoothquant_layers(
        model=model,
        model_batches=batches,
        layer_indices=[0],
        act_quant="per_token",
        act_quant_mode="shared_input",
        smoothquant_alpha=0.5,
        smooth_fold=True,
    )

    layer = model.model.layers[0]
    assert summaries[0].replaced_linears == 7
    assert summaries[0].shared_mlp_modules == 1
    assert isinstance(layer.self_attn.q_proj, SmoothQuantFakeQuantLinear)
    assert isinstance(layer.self_attn.o_proj, SmoothQuantFakeQuantLinear)
    assert isinstance(layer.mlp.gate_proj, SmoothQuantFakeQuantLinear)
    assert isinstance(layer.mlp.up_proj, SmoothQuantFakeQuantLinear)
    assert isinstance(layer.mlp.down_proj, SmoothQuantFakeQuantLinear)
    assert layer.self_attn.q_proj.input_scale is None
    assert layer.self_attn.k_proj.input_scale is None
    assert layer.self_attn.v_proj.input_scale is None
    assert layer.mlp.gate_proj.input_scale is None
    assert layer.mlp.up_proj.input_scale is None
    assert layer.mlp.down_proj.input_scale is None
