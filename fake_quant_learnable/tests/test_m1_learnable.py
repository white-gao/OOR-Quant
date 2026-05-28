from __future__ import annotations

import copy

import torch
import torch.nn as nn

from fake_quant_learnable.apply import (
    apply_baseline_w8a8,
    apply_learned_quant_params,
    apply_learnable_lwt,
    export_learned_quant_params,
    freeze_learnable_lwt,
    iter_learnable_lwt_modules,
    learnable_lwt_parameters,
)
from fake_quant_learnable.calibrate_m1_lwt import calibrate_block_mse
from fake_quant_learnable.modules import (
    BaselineFakeQuantLinear,
    FrozenLearnedFakeQuantLinear,
    LearnableFakeQuantLinear,
)
from fake_quant_learnable.quant import FP8_MAX, fp8_e4m3_qdq_forward, lwt_weight_qdq_ste


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
        self.lm_head = nn.Linear(4, 4)


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)


class TinyQwen3MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TinyQwen3MLPBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = TinyQwen3MLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TinyQwenLikeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = TinyAttention()
        self.mlp = TinyMLP()


def test_apply_learnable_lwt_replaces_linears_and_skips_lm_head() -> None:
    model = TinyModel()

    summary = apply_learnable_lwt(model, act_quant="per_token")

    assert summary.replaced_linears == 2
    assert summary.skipped_linears == 1
    assert isinstance(model.backbone[0], LearnableFakeQuantLinear)
    assert isinstance(model.backbone[2], LearnableFakeQuantLinear)
    assert isinstance(model.lm_head, nn.Linear)
    assert len(list(iter_learnable_lwt_modules(model))) == 2


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


def test_lwt_weight_qdq_ste_forward_uses_fp8_e4m3_after_symmetric_clip() -> None:
    weight = torch.tensor(
        [
            [-2.0, -0.70, -0.12, 0.93],
            [0.18, 0.74, 1.86, -3.0],
        ],
        dtype=torch.float32,
    )
    clip = torch.tensor([[0.75], [1.25]], dtype=torch.float32)
    clipped = torch.minimum(torch.maximum(weight, -clip), clip)
    expected = (
        torch.clamp(clipped / (clip / FP8_MAX), min=-FP8_MAX, max=FP8_MAX)
        .to(torch.float8_e4m3fn)
        .float()
        * (clip / FP8_MAX)
    )

    actual = lwt_weight_qdq_ste(weight, clip)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_apply_baseline_w8a8_replaces_linears_skips_lm_head_and_has_no_parameters() -> None:
    model = TinyModel()

    summary = apply_baseline_w8a8(model, act_quant="per_token")

    assert summary.replaced_linears == 2
    assert summary.skipped_linears == 1
    assert isinstance(model.backbone[0], BaselineFakeQuantLinear)
    assert isinstance(model.backbone[2], BaselineFakeQuantLinear)
    assert isinstance(model.lm_head, nn.Linear)
    for module in model.modules():
        if isinstance(module, BaselineFakeQuantLinear):
            assert list(module.parameters()) == []


def test_let_clip_base_tracks_scaled_weight_distribution() -> None:
    linear = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [0.1, 6.0, 0.2],
                ]
            )
        )
    module = LearnableFakeQuantLinear(linear, act_quant="none", enable_let=True)
    let_scale = torch.tensor([5.0, 0.5, 0.25])
    with torch.no_grad():
        module.log_let_scale.copy_(torch.log(let_scale))

    expected = (linear.weight.float() * let_scale.view(1, -1)).abs().amax(dim=1, keepdim=True)

    torch.testing.assert_close(module.clip, expected, rtol=0, atol=0)


def test_shared_input_mode_patches_qwen_like_mlp_and_runs() -> None:
    torch.manual_seed(11)
    block = TinyQwen3MLPBlock().eval()

    summary = apply_learnable_lwt(
        block,
        act_quant="per_token",
        act_quant_mode="shared_input",
        enable_let=True,
    )

    assert summary.replaced_linears == 3
    assert summary.shared_mlp_modules == 1
    assert block.mlp.gate_proj.log_let_scale is block.mlp.up_proj.log_let_scale
    output = block(torch.randn(2, 4))
    assert output.shape == (2, 4)


def test_apply_learnable_lwt_shares_known_let_input_groups() -> None:
    block = TinyQwenLikeBlock()

    apply_learnable_lwt(block, act_quant="per_token", enable_let=True)

    q = block.self_attn.q_proj
    k = block.self_attn.k_proj
    v = block.self_attn.v_proj
    gate = block.mlp.gate_proj
    up = block.mlp.up_proj
    assert q.log_let_scale is k.log_let_scale is v.log_let_scale
    assert gate.log_let_scale is up.log_let_scale
    assert q.log_let_scale is not gate.log_let_scale
    assert len(list(learnable_lwt_parameters(block))) == 7


def test_learnable_lwt_parameters_can_filter_lwt_and_let() -> None:
    block = TinyQwenLikeBlock()

    apply_learnable_lwt(block, act_quant="per_token", enable_let=True)

    lwt_params = list(learnable_lwt_parameters(block, include_lwt=True, include_let=False))
    let_params = list(learnable_lwt_parameters(block, include_lwt=False, include_let=True))
    all_params = list(learnable_lwt_parameters(block))

    assert len(lwt_params) == 5
    assert len(let_params) == 2
    assert len(all_params) == 7
    assert {id(param) for param in lwt_params}.isdisjoint({id(param) for param in let_params})


def test_let_parameters_get_gradient_and_freeze_matches_forward() -> None:
    torch.manual_seed(3)
    linear = nn.Linear(4, 3, bias=True)
    module = LearnableFakeQuantLinear(
        linear,
        act_quant="per_token",
        init_clip_multiplier=0.8,
        enable_let=True,
    )
    with torch.no_grad():
        module.log_let_scale.copy_(torch.tensor([-0.30, -0.10, 0.20, 0.35]))
    x = torch.randn(5, 4)

    loss = module(x).pow(2).mean()
    loss.backward()

    assert module.log_let_scale.grad is not None
    assert module.log_let_scale.grad.abs().sum() > 0

    module.zero_grad(set_to_none=True)
    with torch.no_grad():
        expected = module(x)
        frozen = module.to_frozen()
        actual = frozen(x)

    assert isinstance(frozen, FrozenLearnedFakeQuantLinear)
    assert frozen.let_scale is not None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_exported_learned_quant_params_recreate_frozen_block_forward() -> None:
    torch.manual_seed(5)
    base = TinyBlock().eval()
    learned = copy.deepcopy(base).eval()
    apply_learnable_lwt(learned, act_quant="per_token", enable_let=True)
    for idx, module in enumerate(iter_learnable_lwt_modules(learned)):
        with torch.no_grad():
            module.log_clip_multiplier.fill_(torch.log(torch.tensor(0.6 + 0.1 * idx)))
            assert module.log_let_scale is not None
            module.log_let_scale.copy_(torch.linspace(-0.2, 0.2, module.in_features))
    params = export_learned_quant_params(learned)

    expected_block = copy.deepcopy(learned).eval()
    freeze_learnable_lwt(expected_block)
    loaded_block = copy.deepcopy(base).eval()
    replaced = apply_learned_quant_params(loaded_block, params)

    x = torch.randn(3, 4)
    assert replaced == 2
    assert isinstance(loaded_block.fc1, FrozenLearnedFakeQuantLinear)
    assert isinstance(loaded_block.fc2, FrozenLearnedFakeQuantLinear)
    torch.testing.assert_close(loaded_block(x), expected_block(x), rtol=0, atol=0)


def test_apply_learnable_lwt_enable_let_creates_let_parameters() -> None:
    model = TinyModel()

    apply_learnable_lwt(model, act_quant="per_token", enable_let=True)

    modules = list(iter_learnable_lwt_modules(model))
    assert len(modules) == 2
    assert all(module.log_let_scale is not None for module in modules)


def test_lwt_parameters_get_scale_gradient_at_minmax_initialization() -> None:
    torch.manual_seed(2)
    linear = nn.Linear(4, 3, bias=False)
    module = LearnableFakeQuantLinear(
        linear,
        act_quant="none",
        init_clip_multiplier=1.0,
    )
    x = torch.randn(7, 4)

    loss = module(x).pow(2).mean()
    loss.backward()

    assert module.log_clip_multiplier.grad is not None
    assert module.log_clip_multiplier.grad.abs().sum() > 0


def test_lwt_parameters_get_gradient_without_training_raw_weight() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(4, 3, bias=False)
    module = LearnableFakeQuantLinear(
        linear,
        act_quant="per_token",
        init_clip_multiplier=0.25,
    )
    x = torch.randn(5, 4)

    loss = module(x).pow(2).mean()
    loss.backward()

    assert module.log_clip_multiplier.grad is not None
    assert torch.isfinite(module.log_clip_multiplier.grad).all()
    assert not any(name == "weight" for name, _ in module.named_parameters())


def test_calibrate_block_mse_reduces_toy_block_reconstruction_loss() -> None:
    torch.manual_seed(1)
    teacher = TinyBlock().eval()
    quant_block = copy.deepcopy(teacher).eval()
    apply_learnable_lwt(
        quant_block,
        act_quant="per_token",
        init_clip_multiplier=0.20,
    )
    batches = [torch.randn(6, 4) for _ in range(4)]

    history = calibrate_block_mse(
        teacher_block=teacher,
        quant_block=quant_block,
        batches=batches,
        steps=25,
        lr=0.05,
    )

    assert history.initial_loss > 0
    assert history.final_loss < history.initial_loss
    assert len(history.losses) == 25
