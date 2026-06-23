from __future__ import annotations

import torch
import torch.nn as nn

from real_quant.naive_w8a8.apply import apply_naive_w8a8
from real_quant.naive_w8a8.modules import RealFP8Linear
from real_quant.naive_w8a8.run_hf_naive_w8a8 import parse_args


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
    assert not hasattr(args, "act_quant_mode")


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

