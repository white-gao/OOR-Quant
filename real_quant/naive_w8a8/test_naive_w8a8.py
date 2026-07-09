from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from real_quant.naive_w8a8 import gptq_runtime
from real_quant.naive_w8a8.apply import _combined_w8a16_forward, apply_naive_w8a8
from real_quant.naive_w8a8.modules import RealFP8Linear
from real_quant.naive_w8a8.run_hf_naive_w8a8 import _run_generation_with_optional_profiler, parse_args
from fake_quant_learnable.gptq import gptq_fp8_quantize_weight


def _cuda_device_with_free_memory(min_free_bytes: int = 128 * 1024 * 1024) -> torch.device | None:
    if not torch.cuda.is_available():
        return None
    best_index = None
    best_free = -1
    for index in range(torch.cuda.device_count()):
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
        except Exception:
            continue
        if free_bytes > best_free:
            best_index = index
            best_free = int(free_bytes)
    if best_index is None or best_free < min_free_bytes:
        return None
    return torch.device(f"cuda:{best_index}")


def test_real_fp8_linear_stores_column_major_weight_and_channel_scales() -> None:
    linear = nn.Linear(16, 32, bias=True, dtype=torch.bfloat16)

    module = RealFP8Linear.from_linear(linear)

    assert module.in_features == 16
    assert module.out_features == 32
    assert module.weight_fp8_t.shape == (16, 32)
    assert module.weight_fp8_t.dtype is torch.float8_e4m3fn
    assert module.weight_fp8_t.stride(0) == 1
    assert module.weight_scale.shape == (1, 32)
    assert module.bias is not None
    assert list(module.parameters()) == []


def test_real_fp8_linear_from_gptq_linear_matches_fake_gptq_qdq_weight() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(8, 12, bias=False, dtype=torch.bfloat16)
    x = torch.randn(5, 8, dtype=torch.float32)
    hessian = x.t().matmul(x) / float(x.shape[0])

    module = RealFP8Linear.from_gptq_linear(linear, hessian, block_size=4)

    expected = gptq_fp8_quantize_weight(linear.weight.detach(), hessian, block_size=4).float()
    actual = (module.weight_fp8_t.float() * module.weight_scale.float()).t().float()
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=1e-3)


def test_apply_naive_w8a8_uses_gptq_hessians_when_provided() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 12, bias=False))
    original_weight = model[0].weight.detach().clone()
    x = torch.randn(5, 8, dtype=torch.float32)
    hessian = x.t().matmul(x) / float(x.shape[0])

    summary = apply_naive_w8a8(model, skip_module_names=(), gptq_hessians={"0": hessian}, gptq_block_size=4)

    assert summary.replaced_linears == 1
    assert isinstance(model[0], RealFP8Linear)
    expected = gptq_fp8_quantize_weight(original_weight, hessian, block_size=4).float()
    actual = model[0].weight_qdq.float()
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=1e-3)


def test_apply_naive_w8a8_replaces_linears_and_skips_lm_head() -> None:
    class TinyModelWithHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 32))
            self.lm_head = nn.Linear(32, 64)

    model = TinyModelWithHead()

    summary = apply_naive_w8a8(model)

    assert summary.replaced_linears == 2
    assert summary.skipped_linears == 1
    assert isinstance(model.backbone[0], RealFP8Linear)
    assert isinstance(model.backbone[2], RealFP8Linear)
    assert isinstance(model.lm_head, nn.Linear)


def test_parse_args_defaults_to_batch_size_one(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_hf_naive_w8a8.py"])

    args = parse_args()

    assert args.batch_size == 1
    assert args.output_dir == "real_quant/naive_w8a8/results"
    assert args.activation_quant_mode == "dynamic"
    assert args.static_activation_calib_samples == 0
    assert args.weight_quant_mode == "minmax"
    assert args.activation_tail_tokens == 0
    assert args.gptq_calib_sample_size == "1024"
    assert args.gptq_calib_split is None
    assert not args.decode_a16_single_token


def test_parse_args_accepts_grad_weighted_gptq(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_hf_naive_w8a8.py",
            "--weight_quant_mode",
            "grad_weighted_gptq",
            "--grad_weight_clip_percentile",
            "95",
            "--grad_weight_floor",
            "0.2",
            "--no-grad_weight_normalize_mean",
        ],
    )

    args = parse_args()

    assert args.weight_quant_mode == "grad_weighted_gptq"
    assert args.grad_weight_clip_percentile == 95.0
    assert args.grad_weight_floor == 0.2
    assert args.grad_weight_normalize_mean is False


def test_parse_args_accepts_slot_grad_weighted_gptq(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_hf_naive_w8a8.py",
            "--weight_quant_mode",
            "slot_grad_weighted_gptq",
            "--grad_weight_clip_percentile",
            "97",
            "--grad_weight_loss_mode",
            "full_sid_multi_target",
            "--grad_weight_max_targets",
            "4",
        ],
    )

    args = parse_args()

    assert args.weight_quant_mode == "slot_grad_weighted_gptq"
    assert args.grad_weight_clip_percentile == 97.0
    assert args.grad_weight_loss_mode == "full_sid_multi_target"
    assert args.grad_weight_max_targets == 4


def test_parse_args_accepts_slot_weighted_gptq_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_hf_naive_w8a8.py",
            "--weight_quant_mode",
            "slot_weighted_gptq",
        ],
    )

    args = parse_args()

    assert args.weight_quant_mode == "slot_weighted_gptq"
    assert args.slot_weight_text == 10.0
    assert args.slot_weight_sid_a == 5.0
    assert args.slot_weight_sid_b == 2.0
    assert args.slot_weight_sid_c == 2.0
    assert args.slot_weight_boundary == 2.0
    assert args.slot_weight_normalize_mean is True


def test_run_generation_with_profiler_exports_chrome_trace(monkeypatch, tmp_path) -> None:
    class FakeLatencyRecord:
        generate_time = 0.001

    class FakeGenerator:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.latency_records = {}

        def generate(self, prompts, **_kwargs):
            self.latency_records = {"sample": FakeLatencyRecord()}
            return {"sample": ["<s_a_1>"]}, None

    class FakeProfiler:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return False

        def key_averages(self):
            return []

        def export_chrome_trace(self, path):
            Path(path).write_text('{"traceEvents": []}', encoding="utf-8")

    monkeypatch.setattr(
        "real_quant.naive_w8a8.run_hf_naive_w8a8.torch.profiler.profile",
        lambda **_kwargs: FakeProfiler(),
    )
    trace_path = tmp_path / "trace.json"

    generations, summary = _run_generation_with_optional_profiler(
        FakeGenerator(),
        {"sample": "prompt"},
        generation_kwargs={},
        profile_fp8=True,
        profile_trace_output=trace_path,
    )

    assert generations == {"sample": ["<s_a_1>"]}
    assert trace_path.read_text(encoding="utf-8") == '{"traceEvents": []}'
    assert summary is not None
    assert summary["chrome_trace_path"] == str(trace_path)


def test_parse_args_defaults_to_first_sid_gradient_loss(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_hf_naive_w8a8.py"])

    args = parse_args()

    assert args.grad_weight_loss_mode == "first_sid"
    assert args.grad_weight_max_targets == 1


def test_build_first_sid_target_token_ids_extracts_first_sid_token() -> None:
    class ToyTokenizer:
        def __call__(self, text, **_kwargs):
            assert text == "<s_a_10><s_b_20><s_c_30>"
            return {"input_ids": torch.tensor([[101, 202, 303]])}

    samples = [{"ground_truth": "<|sid_begin|><s_a_10><s_b_20><s_c_30><|sid_end|>"}]

    target_ids = gptq_runtime.build_first_sid_target_token_ids(ToyTokenizer(), samples)

    assert len(target_ids) == 1
    assert int(target_ids[0]) == 101


def test_build_sid_teacher_forcing_target_token_ids_extracts_multiple_full_sids() -> None:
    class ToyTokenizer:
        mapping = {
            "<s_a_1><s_b_2><s_c_3>": [1, 2, 3],
            "<s_a_4><s_b_5><s_c_6>": [4, 5, 6],
        }

        def __call__(self, text, **_kwargs):
            return {"input_ids": torch.tensor([self.mapping[text]])}

    samples = [
        {
            "ground_truth": (
                "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>"
                "<|sid_begin|><s_a_4><s_b_5><s_c_6><|sid_end|>"
            )
        }
    ]

    target_ids = gptq_runtime.build_sid_teacher_forcing_target_token_ids(
        ToyTokenizer(),
        samples,
        max_items=2,
    )

    assert len(target_ids) == 1
    assert [ids.tolist() for ids in target_ids[0]] == [[1, 2, 3], [4, 5, 6]]


def test_apply_gptq_real_w8a8_layers_uses_layer_specific_token_weights(monkeypatch) -> None:
    class TinyLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)

        def forward(self, hidden_states):
            return self.proj(hidden_states)

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([TinyLayer(), TinyLayer()])

        def forward(self, input_ids, **_kwargs):
            hidden = input_ids.float()
            for layer in self.layers:
                hidden = layer(hidden)
            return hidden

    model = TinyModel()
    model_batches = [{"input_ids": torch.ones(1, 3, 2)}]
    layer_weights = {
        0: [torch.full((1, 3), 2.0)],
        1: [torch.full((1, 3), 3.0)],
    }
    seen = []

    def fake_collect(_layer, _batches, *, token_weight_batches=None):
        seen.append(float(token_weight_batches[0].mean().item()))
        return {"proj": torch.eye(2)}

    def fake_apply_naive_w8a8(*_args, **_kwargs):
        return gptq_runtime.NaiveW8A8Summary(
            replaced_linears=1,
            skipped_linears=0,
            shared_attention_modules=0,
            shared_mlp_modules=0,
        )

    monkeypatch.setattr(gptq_runtime, "collect_gptq_hessians", fake_collect)
    monkeypatch.setattr(gptq_runtime, "apply_naive_w8a8", fake_apply_naive_w8a8)

    summary = gptq_runtime.apply_gptq_real_w8a8_layers(
        model=model,
        model_batches=model_batches,
        layer_indices=[0, 1],
        output_dtype=torch.bfloat16,
        target_regex=None,
        skip_regex=None,
        use_fast_accum=False,
        activation_quant_mode="dynamic",
        decode_a16_when_single_token=True,
        activation_tail_tokens=0,
        damp_percent=0.01,
        block_size=128,
        token_weight_batches_by_layer=layer_weights,
    )

    assert seen == [2.0, 3.0]
    assert summary.replaced_linears == 2


def test_real_fp8_linear_static_activation_uses_buffer_scale() -> None:
    linear = nn.Linear(16, 32, bias=False, dtype=torch.bfloat16)
    module = RealFP8Linear.from_linear(linear)
    module.set_static_activation_scale(torch.tensor(0.25))
    module.set_activation_quant_mode("static")
    x = torch.randn(2, 3, 16, dtype=torch.bfloat16)

    prepared = module.prepare_input(x)

    assert prepared.scale.shape == (6, 1)
    assert torch.allclose(prepared.scale.float(), torch.full((6, 1), 0.25))


def test_real_fp8_linear_reuses_matching_dtype_tensors_without_to_copy() -> None:
    linear = nn.Linear(16, 32, bias=True, dtype=torch.bfloat16)
    module = RealFP8Linear.from_linear(linear)
    x = torch.randn(2, 16, dtype=torch.bfloat16)

    assert module._input_for_output_dtype(x) is x
    assert module._bias_for_output(device=x.device) is module.bias


def test_combined_w8a16_forward_matches_individual_tail_linears() -> None:
    torch.manual_seed(0)
    linears = [
        nn.Linear(16, 24, bias=True, dtype=torch.bfloat16),
        nn.Linear(16, 12, bias=False, dtype=torch.bfloat16),
        nn.Linear(16, 20, bias=True, dtype=torch.bfloat16),
    ]
    modules = tuple(RealFP8Linear.from_linear(linear) for linear in linears)
    x = torch.randn(2, 1, 16, dtype=torch.bfloat16)

    outputs = _combined_w8a16_forward(modules, x)
    expected = tuple(module.reference_w8a16_forward(x) for module in modules)

    assert len(outputs) == len(expected)
    for actual, ref in zip(outputs, expected):
        torch.testing.assert_close(actual.float(), ref.float(), rtol=0, atol=0)


def test_real_fp8_linear_forward_matches_qdq_reference_on_cuda() -> None:
    if not torch.cuda.is_available():
        return
    if not hasattr(torch, "_scaled_mm") or not hasattr(torch, "float8_e4m3fn"):
        return

    device = _cuda_device_with_free_memory()
    if device is None:
        return
    linear = nn.Linear(16, 32, bias=True, dtype=torch.bfloat16)
    x = torch.randn(2, 3, 16, device=device, dtype=torch.bfloat16)
    module = RealFP8Linear.from_linear(linear).to(device)

    assert module.weight_fp8_t.stride(0) == 1

    y = module(x)
    ref = module.reference_qdq_forward(x)

    assert y.shape == ref.shape
    torch.testing.assert_close(y.float(), ref.float(), rtol=2e-2, atol=2e-2)

def test_real_fp8_linear_static_activation_forward_runs_on_cuda() -> None:
    if not torch.cuda.is_available():
        return
    if not hasattr(torch, "_scaled_mm") or not hasattr(torch, "float8_e4m3fn"):
        return

    device = _cuda_device_with_free_memory()
    if device is None:
        return
    linear = nn.Linear(16, 32, bias=True, dtype=torch.bfloat16)
    x = torch.randn(2, 3, 16, device=device, dtype=torch.bfloat16)
    module = RealFP8Linear.from_linear(linear).to(device)
    module.set_static_activation_scale(torch.tensor(0.25))
    module.set_activation_quant_mode("static")

    y = module(x)

    assert y.shape == (2, 3, 32)
    assert y.dtype is torch.bfloat16

def test_real_fp8_linear_decode_a16_uses_qdq_weight_without_activation_quant(monkeypatch) -> None:
    linear = nn.Linear(16, 32, bias=True, dtype=torch.bfloat16)
    module = RealFP8Linear.from_linear(linear, decode_a16_when_single_token=True)
    x_decode = torch.randn(2, 1, 16, dtype=torch.bfloat16)
    x_prefill = torch.randn(2, 3, 16, dtype=torch.bfloat16)

    def fail_prepare(_x):
        raise AssertionError("decode A16 should bypass activation FP8 prepare")

    monkeypatch.setattr(module, "prepare_input", fail_prepare)

    y = module(x_decode)
    ref = module.reference_w8a16_forward(x_decode)

    assert module.should_use_decode_a16(x_decode)
    assert not module.should_use_decode_a16(x_prefill)
    assert y.shape == (2, 1, 32)
    torch.testing.assert_close(y.float(), ref.float(), rtol=0, atol=0)


def test_real_fp8_linear_tail1_keeps_prefill_last_token_in_a16() -> None:
    linear = nn.Linear(16, 32, bias=True, dtype=torch.bfloat16)
    module = RealFP8Linear.from_linear(
        linear,
        activation_quant_mode="static",
        activation_tail_tokens=1,
    )
    module.set_static_activation_scale(torch.tensor(0.25))
    x = torch.randn(2, 3, 16, dtype=torch.bfloat16)

    y = module(x)
    main = module.forward_prepared(module.prepare_input(x[..., :-1, :]))
    tail = module.reference_w8a16_forward(x[..., -1:, :])
    ref = torch.cat([main, tail], dim=-2)

    assert y.shape == (2, 3, 32)
    torch.testing.assert_close(y.float(), ref.float(), rtol=0, atol=0)


def test_tail1_shared_mlp_preserves_gate_up_shared_prepare(monkeypatch) -> None:
    model = TinySharedMLP()
    summary = apply_naive_w8a8(
        model,
        activation_quant_mode="static",
        activation_tail_tokens=1,
    )
    x = torch.randn(2, 3, 16, dtype=torch.bfloat16)

    calls = []
    original_prepare = RealFP8Linear.prepare_input

    def counted_prepare(self, hidden_states):
        calls.append(self.out_features)
        return original_prepare(self, hidden_states)

    monkeypatch.setattr(RealFP8Linear, "prepare_input", counted_prepare)

    y = model(x)

    assert y.shape == (2, 3, 16)
    assert summary.shared_mlp_modules == 1
    assert calls == [32, 16]


def test_tail1_shared_mlp_combines_gate_up_tail_w8a16(monkeypatch) -> None:
    model = TinySharedMLP()
    apply_naive_w8a8(
        model,
        activation_quant_mode="static",
        activation_tail_tokens=1,
    )
    x = torch.randn(2, 3, 16, dtype=torch.bfloat16)

    calls = []
    original_forward_w8a16 = RealFP8Linear.forward_w8a16

    def counted_forward_w8a16(self, hidden_states):
        calls.append(self.out_features)
        return original_forward_w8a16(self, hidden_states)

    monkeypatch.setattr(RealFP8Linear, "forward_w8a16", counted_forward_w8a16)

    y = model(x)

    assert y.shape == (2, 3, 16)
    assert calls == [16]

class TinySharedMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(16, 32, bias=False)
        self.up_proj = nn.Linear(16, 32, bias=False)
        self.down_proj = nn.Linear(32, 16, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def test_real_fp8_linear_can_reuse_prepared_activation_on_cuda() -> None:
    if not torch.cuda.is_available():
        return
    if not hasattr(torch, "_scaled_mm") or not hasattr(torch, "float8_e4m3fn"):
        return

    device = _cuda_device_with_free_memory()
    if device is None:
        return
    linear = nn.Linear(16, 32, bias=True, dtype=torch.bfloat16)
    module = RealFP8Linear.from_linear(linear).to(device)
    x = torch.randn(2, 3, 16, device=device, dtype=torch.bfloat16)

    prepared = module.prepare_input(x)
    y_prepared = module.forward_prepared(prepared)
    y_direct = module(x)

    assert prepared.x_fp8.dtype is torch.float8_e4m3fn
    assert prepared.scale.shape == (6, 1)
    torch.testing.assert_close(y_prepared.float(), y_direct.float(), rtol=0, atol=0)


def test_shared_input_mode_patches_mlp_and_quantizes_gate_up_once(monkeypatch) -> None:
    if not torch.cuda.is_available():
        return
    if not hasattr(torch, "_scaled_mm") or not hasattr(torch, "float8_e4m3fn"):
        return

    device = _cuda_device_with_free_memory()
    if device is None:
        return
    model = TinySharedMLP()
    summary = apply_naive_w8a8(model)
    x = torch.randn(2, 3, 16, device=device, dtype=torch.bfloat16)
    model.to(device)

    calls = []
    original_prepare = RealFP8Linear.prepare_input

    def counted_prepare(self, hidden_states):
        calls.append(self.out_features)
        return original_prepare(self, hidden_states)

    monkeypatch.setattr(RealFP8Linear, "prepare_input", counted_prepare)

    y = model(x)

    assert y.shape == (2, 3, 16)
    assert summary.shared_mlp_modules == 1
    assert calls == [32, 16]


def test_decode_a16_shared_mlp_bypasses_shared_activation_prepare(monkeypatch) -> None:
    model = TinySharedMLP()
    summary = apply_naive_w8a8(model, decode_a16_when_single_token=True)
    x = torch.randn(2, 1, 16, dtype=torch.bfloat16)

    calls = []
    original_prepare = RealFP8Linear.prepare_input

    def counted_prepare(self, hidden_states):
        calls.append(self.out_features)
        return original_prepare(self, hidden_states)

    monkeypatch.setattr(RealFP8Linear, "prepare_input", counted_prepare)

    y = model(x)

    assert y.shape == (2, 1, 16)
    assert summary.shared_mlp_modules == 1
    assert calls == []

