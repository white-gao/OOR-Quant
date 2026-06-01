#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .apply import (
    BaselineQuantSummary,
    SmoothScope,
    apply_baseline_w8a8,
    apply_learnable_lwt,
    export_learned_quant_params,
    freeze_learnable_lwt,
    install_shared_input_activation_quantization,
    learned_quantized_module_from_params,
    should_apply_smooth_transform,
)
from .calibrate_m1_lwt import (
    DEFAULT_EPOCHS,
    DEFAULT_LET_LR,
    DEFAULT_LWT_LR,
    Batch,
    CalibrationHistory,
    _batch_to_args_kwargs,
    _first_tensor,
    calibrate_block_mse,
)
from .modules import BaselineFakeQuantLinear, FrozenLearnedFakeQuantLinear, LearnableFakeQuantLinear
from .quant import ActQuant, ActQuantMode, fp8_weight_per_channel_forward
from fake_quant.smoothquant.core import compute_smooth_scale, smooth_linear_weight


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
for path in (PROJECT_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark import Benchmark  # noqa: E402
from benchmark.tasks.v1_0.registry import get_loader, get_task_config  # noqa: E402


DEFAULT_MODEL_PATH = "/home/guowei/OneRec-1.7B/"
DEFAULT_DATA_DIR = "data/onerec_data/benchmark-data-calib1024"
DEFAULT_SMOOTHQUANT_ALPHA = 0.5
DEFAULT_SMOOTHQUANT_MIN_SCALE = None
DEFAULT_SMOOTHQUANT_MAX_SCALE = None
DEFAULT_SMOOTH_SCOPE: SmoothScope = "omni"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value_str = str(value).strip().lower()
    if value_str in {"1", "true", "yes", "y", "on"}:
        return True
    if value_str in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OneRec AD evaluation with baseline, M1 LWT, M2 LET-only, or M2 LWT+LET FP8 W+A fake quantization."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--calib_data_dir", default=None)
    parser.add_argument("--eval_data_dir", default=None)
    parser.add_argument("--output_dir", default="fake_quant_learnable/results/ptq_ad")
    parser.add_argument("--mode", default="m2_lwt_let", choices=["baseline_w8a8", "smoothquant_w8a8", "m1_lwt", "m2_let", "m2_lwt_let"])
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--calib_split", default=None, choices=["test", "calib"])
    parser.add_argument("--calib_sample_size", default="1024")
    parser.add_argument("--calib_offset", type=int, default=0)
    parser.add_argument("--eval_sample_size", default="full")
    parser.add_argument("--eval_offset", type=int, default=0)
    parser.add_argument("--layers", default="all", help='Layer spec: "all", "last:K", "0,2-4".')
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lwt_lr", type=float, default=DEFAULT_LWT_LR, help="Learning rate for LWT clipping parameters.")
    parser.add_argument("--let_lr", type=float, default=DEFAULT_LET_LR, help="Learning rate for LET scale parameters.")
    parser.add_argument("--let_init", default="ones", choices=["ones", "smoothquant"], help="Initialization for LET scale parameters.")
    parser.add_argument("--smoothquant_alpha", type=float, default=DEFAULT_SMOOTHQUANT_ALPHA)
    parser.add_argument(
        "--smooth_scope",
        default=DEFAULT_SMOOTH_SCOPE,
        choices=["all", "omni"],
        help="all smooths every Linear; omni smooths q/k/v, o_proj, and gate/up/down.",
    )
    parser.add_argument(
        "--smooth_fold",
        type=parse_bool,
        default=True,
        help="Fold SmoothQuant/LET activation scaling into adjacent modules when the transform is exact.",
    )
    parser.add_argument(
        "--smoothquant_min_scale",
        type=float,
        default=DEFAULT_SMOOTHQUANT_MIN_SCALE,
        help="Optional SmoothQuant scale lower clamp. Omit to match fake_quant SmoothQuant.",
    )
    parser.add_argument(
        "--smoothquant_max_scale",
        type=float,
        default=DEFAULT_SMOOTHQUANT_MAX_SCALE,
        help="Optional SmoothQuant scale upper clamp. Omit to match fake_quant SmoothQuant.",
    )
    parser.add_argument("--init_clip_multiplier", type=float, default=1.0)
    parser.add_argument("--act_quant", default="per_token", choices=["none", "per_token"])
    parser.add_argument(
        "--act_quant_mode",
        default="shared_input",
        choices=["per_linear", "shared_input"],
        help="per_linear quantizes each Linear input independently; shared_input reuses qkv/gate-up activation QDQ.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--num_beams", type=int, default=32)
    parser.add_argument("--num_return_sequences", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--calib_only", action="store_true")
    parser.add_argument("--save_model_state", action="store_true")
    parser.add_argument("--save_quant_params", dest="save_quant_params", action="store_true", default=True)
    parser.add_argument("--no_save_quant_params", dest="save_quant_params", action="store_false")
    parser.add_argument("--load_quant_params", default=None)
    parser.add_argument("--skip_calibration", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def parse_sample_size(value: Any) -> Any:
    if value is None or value == "":
        return None
    if value == "full":
        return "full"
    return int(value)


def resolve_repo_path(path: str | os.PathLike[str]) -> Path:
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


def default_calib_split(data_dir: str | os.PathLike[str], fallback_split: str) -> str:
    calib_file = resolve_repo_path(data_dir) / "ad" / "ad_calib.parquet"
    return "calib" if calib_file.exists() else fallback_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_input_device(model: nn.Module, fallback: str) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        for device in hf_device_map.values():
            if isinstance(device, str) and device not in {"cpu", "disk"}:
                return torch.device(device)
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise AttributeError("Could not find transformer layers at model.model.layers or model.layers.")
    if not isinstance(layers, nn.ModuleList):
        raise TypeError(f"Expected layers to be nn.ModuleList, got {type(layers)!r}.")
    return layers


def parse_layer_indices(spec: str, *, num_layers: int) -> list[int]:
    spec = spec.strip()
    if spec == "all":
        return list(range(num_layers))
    if spec.startswith("last:"):
        count = int(spec.split(":", 1)[1])
        if count <= 0 or count > num_layers:
            raise ValueError(f"Invalid last layer count: {count}")
        return list(range(num_layers - count, num_layers))

    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid layer range: {part}")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    deduped = sorted(set(indices))
    if not deduped:
        raise ValueError("No layer indices selected.")
    for idx in deduped:
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {idx} out of range for {num_layers} layers.")
    return deduped


def load_ad_data(
    tokenizer: Any,
    data_dir: str,
    split: str,
    sample_size: Any,
    sample_offset: int = 0,
) -> dict[str, dict[str, Any]]:
    if sample_offset < 0:
        raise ValueError(f"sample_offset must be non-negative, got {sample_offset}")

    loader_sample_size = sample_size
    if isinstance(sample_size, int):
        loader_sample_size = sample_size + sample_offset

    loader = get_loader(
        task_name="ad",
        data_dir=data_dir,
        enable_thinking=False,
        tokenizer=tokenizer,
    )
    data = loader.load_data(split=split, sample_size=loader_sample_size)
    if sample_offset == 0 and not isinstance(sample_size, int):
        return data

    items = list(data.items())
    if sample_offset:
        if sample_offset >= len(items):
            raise ValueError(
                f"sample_offset={sample_offset} leaves no samples after loading {len(items)} rows."
            )
        items = items[sample_offset:]
    if isinstance(sample_size, int):
        items = items[:sample_size]
    return dict(items)


def format_prompt(prompt: str, prompt_token: str) -> str:
    if prompt_token and not prompt.endswith(prompt_token):
        return prompt + prompt_token
    return prompt


def build_model_batches(
    *,
    tokenizer: Any,
    prompts: Sequence[str],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        batches.append(
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in encoded.items()
            }
        )
    return batches


def capture_layer_input_batches(
    *,
    model: nn.Module,
    layer: nn.Module,
    model_batches: Iterable[Mapping[str, Any]],
) -> list[Batch]:
    captured: list[Batch] = []

    def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        captured.append((_detach_tree(args), _detach_tree(kwargs)))

    handle = layer.register_forward_pre_hook(hook, with_kwargs=True)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in model_batches:
                try:
                    model(**batch, use_cache=False)
                except TypeError:
                    model(**batch)
    finally:
        handle.remove()
        model.train(was_training)
    return captured


def advance_layer_input_batches(
    *,
    layer: nn.Module,
    batches: Sequence[Batch],
) -> list[Batch]:
    """Run one transformer block and build input batches for the next block."""
    advanced: list[Batch] = []
    was_training = layer.training
    target_device = _module_device(layer)
    layer.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                args, kwargs = _batch_to_args_kwargs(batch)
                args = _move_tree_to_device(args, target_device)
                kwargs = _move_tree_to_device(kwargs, target_device)
                output = layer(*args, **kwargs)
                hidden = _first_tensor(output).detach()
                next_args: tuple[Any, ...] = (hidden, *args[1:]) if args else ()
                next_kwargs = dict(kwargs)
                if not next_args:
                    if "hidden_states" in next_kwargs:
                        next_kwargs["hidden_states"] = hidden
                    else:
                        next_args = (hidden,)
                advanced.append((_detach_tree(next_args), _detach_tree(next_kwargs)))
    finally:
        layer.train(was_training)
    return advanced


def collect_smoothquant_scales(
    module: nn.Module,
    batches: Sequence[Batch],
    *,
    alpha: float = DEFAULT_SMOOTHQUANT_ALPHA,
    min_scale: float | None = DEFAULT_SMOOTHQUANT_MIN_SCALE,
    max_scale: float | None = DEFAULT_SMOOTHQUANT_MAX_SCALE,
    smooth_scope: SmoothScope = DEFAULT_SMOOTH_SCOPE,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"smoothquant alpha must be in [0, 1], got {alpha}")
    linear_modules = {
        name: child
        for name, child in module.named_modules()
        if isinstance(child, nn.Linear) and should_apply_smooth_transform(name, smooth_scope)
    }
    if isinstance(module, nn.Linear):
        linear_modules = {"": module}
    if not linear_modules:
        return {}

    act_absmax = _collect_linear_input_absmax(module, batches, linear_modules)
    scales: dict[str, torch.Tensor] = {}
    grouped: set[str] = set()
    for group in _known_smoothquant_input_group_names(linear_modules):
        members = [name for name in group if name in act_absmax]
        if len(members) != len(group):
            continue
        act_max = torch.stack([act_absmax[name].float().cpu() for name in members]).amax(dim=0)
        weight_max = torch.stack([
            _linear_input_weight_absmax(linear_modules[name], eps=eps).cpu()
            for name in members
        ]).amax(dim=0)
        scale = _smoothquant_scale(
            act_max,
            weight_max,
            alpha=alpha,
            min_scale=min_scale,
            max_scale=max_scale,
            eps=eps,
        )
        for name in members:
            scales[name] = scale.clone()
            grouped.add(name)

    for name, linear in linear_modules.items():
        if name in grouped or name not in act_absmax:
            continue
        scales[name] = _smoothquant_scale(
            act_absmax[name].float().cpu(),
            _linear_input_weight_absmax(linear, eps=eps).cpu(),
            alpha=alpha,
            min_scale=min_scale,
            max_scale=max_scale,
            eps=eps,
        )
    return scales


def _collect_linear_input_absmax(
    module: nn.Module,
    batches: Sequence[Batch],
    linear_modules: Mapping[str, nn.Linear],
) -> dict[str, torch.Tensor]:
    stats: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str, linear: nn.Linear):
        def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if args:
                x = args[0]
            else:
                x = kwargs.get("input", kwargs.get("hidden_states"))
            if not torch.is_tensor(x):
                return
            if x.shape[-1] != linear.in_features:
                raise ValueError(
                    f"Expected input last dim {linear.in_features} for {name!r}, got {tuple(x.shape)}"
                )
            reduce_dims = tuple(range(x.ndim - 1))
            current = x.detach().float().abs()
            current = current.amax(dim=reduce_dims) if reduce_dims else current
            previous = stats.get(name)
            stats[name] = current.cpu() if previous is None else torch.maximum(previous, current.cpu())
        return hook

    for name, linear in linear_modules.items():
        handles.append(linear.register_forward_pre_hook(make_hook(name, linear), with_kwargs=True))

    was_training = module.training
    target_device = _module_device(module)
    module.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                args, kwargs = _batch_to_args_kwargs(batch)
                args = _move_tree_to_device(args, target_device)
                kwargs = _move_tree_to_device(kwargs, target_device)
                module(*args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
        module.train(was_training)
    return stats


def _linear_input_weight_absmax(linear: nn.Linear, *, eps: float = 1e-12) -> torch.Tensor:
    return linear.weight.detach().float().abs().amax(dim=0).clamp_min(eps)


def _smoothquant_scale(
    act_absmax: torch.Tensor,
    weight_absmax: torch.Tensor,
    *,
    alpha: float,
    min_scale: float | None,
    max_scale: float | None,
    eps: float = 1e-12,
) -> torch.Tensor:
    scale = compute_smooth_scale(act_absmax, weight_absmax, alpha=alpha, eps=eps)
    if min_scale is not None:
        scale = scale.clamp_min(float(min_scale))
    if max_scale is not None:
        scale = scale.clamp_max(float(max_scale))
    return scale


def _known_smoothquant_input_group_names(modules: Mapping[str, nn.Module]) -> list[tuple[str, ...]]:
    groups: list[tuple[str, ...]] = []
    for name in sorted(modules):
        if name.endswith(".q_proj"):
            prefix = name[: -len(".q_proj")]
            group = (f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj")
            if all(member in modules for member in group):
                groups.append(group)
        elif name.endswith(".gate_proj"):
            prefix = name[: -len(".gate_proj")]
            group = (f"{prefix}.gate_proj", f"{prefix}.up_proj")
            if all(member in modules for member in group):
                groups.append(group)
    return groups


def apply_smoothquant_scales_to_learnable(
    module: nn.Module,
    scales: Mapping[str, torch.Tensor],
) -> int:
    applied = 0
    modules = dict(module.named_modules())
    if isinstance(module, LearnableFakeQuantLinear):
        modules = {"": module}
    for name, scale in scales.items():
        target = modules.get(name)
        if not isinstance(target, LearnableFakeQuantLinear) or target.log_let_scale is None:
            continue
        scale_tensor = scale.to(device=target.log_let_scale.device, dtype=target.log_let_scale.dtype).reshape_as(
            target.log_let_scale
        )
        scale_tensor = scale_tensor.clamp(min=target.min_let_scale, max=target.max_let_scale)
        with torch.no_grad():
            target.log_let_scale.copy_(torch.log(scale_tensor))
        applied += 1
    return applied


def smoothquant_quantized_module_from_scales(
    module: nn.Module,
    scales: Mapping[str, torch.Tensor],
    *,
    act_quant: ActQuant,
    smooth_scope: SmoothScope = DEFAULT_SMOOTH_SCOPE,
    folded_names: set[str] | None = None,
) -> tuple[nn.Module, int]:
    if isinstance(module, nn.Linear):
        scale = scales.get("")
        if scale is None:
            raise ValueError("Missing SmoothQuant scale for root Linear module.")
        return _smoothquant_frozen_linear(module, scale, act_quant=act_quant), 1
    return module, _replace_children_smoothquant(
        module,
        scales=scales,
        prefix="",
        act_quant=act_quant,
        smooth_scope=smooth_scope,
        folded_names=folded_names or set(),
    )


def _replace_children_smoothquant(
    module: nn.Module,
    *,
    scales: Mapping[str, torch.Tensor],
    prefix: str,
    act_quant: ActQuant,
    smooth_scope: SmoothScope,
    folded_names: set[str],
) -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear):
            if should_apply_smooth_transform(full_name, smooth_scope):
                scale = scales.get(full_name)
                if scale is None:
                    raise KeyError(f"Missing SmoothQuant scale for Linear module: {full_name}")
                replacement = _smoothquant_frozen_linear(
                    child,
                    scale,
                    act_quant=act_quant,
                    fold_activation=full_name in folded_names,
                )
            else:
                replacement = BaselineFakeQuantLinear(child, act_quant=act_quant)
            setattr(module, child_name, replacement)
            replaced += 1
            continue
        replaced += _replace_children_smoothquant(
            child,
            scales=scales,
            prefix=full_name,
            act_quant=act_quant,
            smooth_scope=smooth_scope,
            folded_names=folded_names,
        )
    return replaced


def _smoothquant_frozen_linear(
    linear: nn.Linear,
    scale: torch.Tensor,
    *,
    act_quant: ActQuant,
    fold_activation: bool = False,
) -> FrozenLearnedFakeQuantLinear:
    scale = scale.detach().float().reshape(-1).to(device=linear.weight.device)
    if scale.numel() != linear.in_features:
        raise ValueError(f"Expected SmoothQuant scale shape ({linear.in_features},), got {tuple(scale.shape)}")
    with torch.no_grad():
        scaled_weight = linear.weight.detach() if fold_activation else smooth_linear_weight(linear.weight.detach(), scale)
        weight_qdq = fp8_weight_per_channel_forward(scaled_weight)
        bias = None if linear.bias is None else linear.bias.detach().clone()
        let_scale = None if fold_activation else scale.detach().cpu()
    return FrozenLearnedFakeQuantLinear(
        weight_qdq=weight_qdq,
        bias=bias,
        act_quant=act_quant,
        let_scale=let_scale,
    )



def fold_smoothquant_scales_inplace(
    module: nn.Module,
    scales: Mapping[str, torch.Tensor],
    *,
    smooth_scope: SmoothScope = DEFAULT_SMOOTH_SCOPE,
) -> set[str]:
    """Fold explicit SmoothQuant activation scaling into adjacent FP modules where exact."""
    folded: set[str] = set()
    if smooth_scope != "omni":
        return folded
    folded.update(
        _fold_norm_to_linear_input_group(
            module,
            norm_name="input_layernorm",
            linear_names=("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
            scales=scales,
        )
    )
    folded.update(
        _fold_norm_to_linear_input_group(
            module,
            norm_name="post_attention_layernorm",
            linear_names=("mlp.gate_proj", "mlp.up_proj"),
            scales=scales,
        )
    )
    if _fold_linear_output_to_linear_input(
        module,
        source_name="self_attn.v_proj",
        target_name="self_attn.o_proj",
        scales=scales,
    ):
        folded.add("self_attn.o_proj")
    if _fold_linear_output_to_linear_input(
        module,
        source_name="mlp.up_proj",
        target_name="mlp.down_proj",
        scales=scales,
    ):
        folded.add("mlp.down_proj")
    return folded


def fold_frozen_let_scales_inplace(
    module: nn.Module,
    *,
    smooth_scope: SmoothScope = DEFAULT_SMOOTH_SCOPE,
) -> set[str]:
    """Fold frozen LET activation scales into adjacent modules after calibration."""
    folded: set[str] = set()
    if smooth_scope != "omni":
        return folded
    folded.update(
        _fold_norm_to_frozen_input_group(
            module,
            norm_name="input_layernorm",
            linear_names=("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        )
    )
    folded.update(
        _fold_norm_to_frozen_input_group(
            module,
            norm_name="post_attention_layernorm",
            linear_names=("mlp.gate_proj", "mlp.up_proj"),
        )
    )
    if _fold_frozen_output_to_frozen_input(
        module,
        source_name="self_attn.v_proj",
        target_name="self_attn.o_proj",
    ):
        folded.add("self_attn.o_proj")
    if _fold_frozen_output_to_frozen_input(
        module,
        source_name="mlp.up_proj",
        target_name="mlp.down_proj",
    ):
        folded.add("mlp.down_proj")
    return folded


def _fold_norm_to_linear_input_group(
    module: nn.Module,
    *,
    norm_name: str,
    linear_names: Sequence[str],
    scales: Mapping[str, torch.Tensor],
) -> set[str]:
    norm = _maybe_get_submodule(module, norm_name)
    if norm is None or not hasattr(norm, "weight"):
        return set()
    scale = _shared_scale_from_mapping(scales, linear_names)
    if scale is None:
        return set()
    linears = [_maybe_get_submodule(module, name) for name in linear_names]
    if not all(isinstance(linear, nn.Linear) for linear in linears):
        return set()
    if not _scale_matches_norm_and_linear_inputs(scale, norm, linears):
        return set()

    with torch.no_grad():
        _divide_weight_or_bias(norm, "weight", scale)
        _divide_weight_or_bias(norm, "bias", scale)
        for linear in linears:
            assert isinstance(linear, nn.Linear)
            linear.weight.mul_(scale.to(device=linear.weight.device, dtype=linear.weight.dtype).view(1, -1))
    return set(linear_names)


def _fold_linear_output_to_linear_input(
    module: nn.Module,
    *,
    source_name: str,
    target_name: str,
    scales: Mapping[str, torch.Tensor],
) -> bool:
    source = _maybe_get_submodule(module, source_name)
    target = _maybe_get_submodule(module, target_name)
    scale = scales.get(target_name)
    if not isinstance(source, nn.Linear) or not isinstance(target, nn.Linear) or scale is None:
        return False
    scale = scale.detach().float().reshape(-1)
    if source.out_features != target.in_features or scale.numel() != target.in_features:
        return False

    with torch.no_grad():
        source.weight.div_(scale.to(device=source.weight.device, dtype=source.weight.dtype).view(-1, 1))
        if source.bias is not None:
            source.bias.div_(scale.to(device=source.bias.device, dtype=source.bias.dtype))
        target.weight.mul_(scale.to(device=target.weight.device, dtype=target.weight.dtype).view(1, -1))
    return True


def _fold_norm_to_frozen_input_group(
    module: nn.Module,
    *,
    norm_name: str,
    linear_names: Sequence[str],
) -> set[str]:
    norm = _maybe_get_submodule(module, norm_name)
    if norm is None or not hasattr(norm, "weight"):
        return set()
    linears = [_maybe_get_submodule(module, name) for name in linear_names]
    if not all(isinstance(linear, FrozenLearnedFakeQuantLinear) for linear in linears):
        return set()
    scale = _shared_scale_from_frozen_linears(linears)
    if scale is None or not _scale_matches_norm_and_linear_inputs(scale, norm, linears):
        return set()

    with torch.no_grad():
        _divide_weight_or_bias(norm, "weight", scale)
        _divide_weight_or_bias(norm, "bias", scale)
    for linear in linears:
        assert isinstance(linear, FrozenLearnedFakeQuantLinear)
        linear.let_scale = None
    return set(linear_names)


def _fold_frozen_output_to_frozen_input(
    module: nn.Module,
    *,
    source_name: str,
    target_name: str,
) -> bool:
    source = _maybe_get_submodule(module, source_name)
    target = _maybe_get_submodule(module, target_name)
    if not isinstance(source, FrozenLearnedFakeQuantLinear) or not isinstance(target, FrozenLearnedFakeQuantLinear):
        return False
    if target.let_scale is None:
        return False
    scale = target.let_scale.detach().float().reshape(-1)
    if source.out_features != target.in_features or scale.numel() != target.in_features:
        return False

    with torch.no_grad():
        source.weight_qdq.div_(scale.to(device=source.weight_qdq.device, dtype=source.weight_qdq.dtype).view(-1, 1))
        if source.bias is not None:
            source.bias.div_(scale.to(device=source.bias.device, dtype=source.bias.dtype))
    target.let_scale = None
    return True


def _shared_scale_from_mapping(
    scales: Mapping[str, torch.Tensor],
    names: Sequence[str],
) -> torch.Tensor | None:
    tensors = [scales.get(name) for name in names]
    if any(tensor is None for tensor in tensors):
        return None
    scale = tensors[0].detach().float().reshape(-1)
    for tensor in tensors[1:]:
        other = tensor.detach().float().reshape(-1)
        if scale.shape != other.shape or not torch.allclose(scale, other, rtol=1e-4, atol=1e-6):
            return None
    return scale


def _shared_scale_from_frozen_linears(modules: Sequence[nn.Module | None]) -> torch.Tensor | None:
    scales = []
    for module in modules:
        if not isinstance(module, FrozenLearnedFakeQuantLinear) or module.let_scale is None:
            return None
        scales.append(module.let_scale.detach().float().reshape(-1))
    scale = scales[0]
    for other in scales[1:]:
        if scale.shape != other.shape or not torch.allclose(scale, other, rtol=1e-4, atol=1e-6):
            return None
    return scale


def _scale_matches_norm_and_linear_inputs(
    scale: torch.Tensor,
    norm: nn.Module,
    linears: Sequence[nn.Module | None],
) -> bool:
    weight = getattr(norm, "weight", None)
    if not torch.is_tensor(weight) or scale.numel() != weight.numel():
        return False
    for linear in linears:
        in_features = getattr(linear, "in_features", None)
        if in_features != scale.numel():
            return False
    return True


def _divide_weight_or_bias(module: nn.Module, attr_name: str, scale: torch.Tensor) -> None:
    value = getattr(module, attr_name, None)
    if not torch.is_tensor(value):
        return
    value.div_(scale.to(device=value.device, dtype=value.dtype).reshape_as(value))


def _maybe_get_submodule(module: nn.Module, name: str) -> nn.Module | None:
    try:
        return module.get_submodule(name)
    except AttributeError:
        return None



def calibrate_model_layers_m1(
    *,
    model: nn.Module,
    model_batches: Sequence[Mapping[str, Any]],
    layer_indices: Sequence[int],
    epochs: int,
    act_quant: ActQuant,
    lwt_lr: float = DEFAULT_LWT_LR,
    let_lr: float = DEFAULT_LET_LR,
    act_quant_mode: ActQuantMode = "per_linear",
    init_clip_multiplier: float = 1.0,
    enable_let: bool = False,
    train_lwt: bool = True,
    let_init: str = "ones",
    smoothquant_alpha: float = DEFAULT_SMOOTHQUANT_ALPHA,
    smoothquant_min_scale: float | None = DEFAULT_SMOOTHQUANT_MIN_SCALE,
    smoothquant_max_scale: float | None = DEFAULT_SMOOTHQUANT_MAX_SCALE,
    smooth_scope: SmoothScope = DEFAULT_SMOOTH_SCOPE,
    smooth_fold: bool = True,
    learned_quant_params: MutableMapping[int, dict[str, Any]] | None = None,
) -> dict[int, CalibrationHistory]:
    layers = get_transformer_layers(model)
    histories: dict[int, CalibrationHistory] = {}
    selected_layer_indices = sorted(layer_indices)
    fp_inputs: list[Batch] | None = None
    quant_inputs: list[Batch] | None = None
    stream_layer_idx: int | None = None

    for layer_idx in selected_layer_indices:
        if fp_inputs is None or quant_inputs is None:
            fp_inputs = capture_layer_input_batches(
                model=model,
                layer=layers[layer_idx],
                model_batches=model_batches,
            )
            quant_inputs = _detach_tree(fp_inputs)
            stream_layer_idx = layer_idx
        else:
            if stream_layer_idx is None:
                raise RuntimeError("Internal error: stream_layer_idx is not initialized.")
            while stream_layer_idx < layer_idx:
                pass_through_layer = layers[stream_layer_idx]
                fp_inputs = advance_layer_input_batches(
                    layer=pass_through_layer,
                    batches=fp_inputs,
                )
                quant_inputs = advance_layer_input_batches(
                    layer=pass_through_layer,
                    batches=quant_inputs,
                )
                stream_layer_idx += 1

        teacher_block = layers[layer_idx]
        quant_block = copy.deepcopy(teacher_block)
        smoothquant_scales: dict[str, torch.Tensor] = {}
        if enable_let and let_init == "smoothquant":
            smoothquant_scales = collect_smoothquant_scales(
                teacher_block,
                fp_inputs,
                alpha=smoothquant_alpha,
                min_scale=smoothquant_min_scale,
                max_scale=smoothquant_max_scale,
                smooth_scope=smooth_scope,
            )
        elif let_init != "ones":
            raise ValueError(f"Unsupported LET initialization: {let_init}")

        if isinstance(quant_block, nn.Linear):
            quant_block = LearnableFakeQuantLinear(
                quant_block,
                act_quant=act_quant,
                init_clip_multiplier=init_clip_multiplier,
                enable_let=enable_let,
            )
        else:
            apply_learnable_lwt(
                quant_block,
                act_quant=act_quant,
                act_quant_mode=act_quant_mode,
                init_clip_multiplier=init_clip_multiplier,
                enable_let=enable_let,
                let_scope=smooth_scope,
            )
        if smoothquant_scales:
            applied_scales = apply_smoothquant_scales_to_learnable(quant_block, smoothquant_scales)
            if applied_scales == 0:
                raise ValueError(f"No SmoothQuant LET scales were applied for layer {layer_idx}.")
        history = calibrate_block_mse(
            teacher_block=teacher_block,
            quant_block=quant_block,
            batches=quant_inputs,
            target_batches=fp_inputs,
            epochs=epochs,
            lwt_lr=lwt_lr,
            let_lr=let_lr,
            train_lwt=train_lwt,
            train_let=enable_let,
        )
        if learned_quant_params is not None:
            learned_quant_params[layer_idx] = export_learned_quant_params(quant_block)
        if isinstance(quant_block, LearnableFakeQuantLinear):
            frozen_quant_block = quant_block.to_frozen()
        else:
            freeze_learnable_lwt(quant_block)
            frozen_quant_block = quant_block
        folded_names = (
            fold_frozen_let_scales_inplace(frozen_quant_block, smooth_scope=smooth_scope)
            if smooth_fold and enable_let
            else set()
        )

        fp_inputs = advance_layer_input_batches(
            layer=teacher_block,
            batches=fp_inputs,
        )
        quant_inputs = advance_layer_input_batches(
            layer=frozen_quant_block,
            batches=quant_inputs,
        )
        stream_layer_idx = layer_idx + 1
        layers[layer_idx] = frozen_quant_block
        histories[layer_idx] = history
        if enable_let:
            label = "M2-LET" if not train_lwt else "M2"
        else:
            label = "M1"
        print(
            f"[{label}] layer={layer_idx} initial_loss={history.initial_loss:.6g} "
            f"final_loss={history.final_loss:.6g} folded={len(folded_names)}"
        )
    return histories


def apply_smoothquant_layers(
    *,
    model: nn.Module,
    model_batches: Sequence[Mapping[str, Any]],
    layer_indices: Sequence[int],
    act_quant: ActQuant,
    act_quant_mode: ActQuantMode = "per_linear",
    smoothquant_alpha: float = DEFAULT_SMOOTHQUANT_ALPHA,
    smoothquant_min_scale: float | None = DEFAULT_SMOOTHQUANT_MIN_SCALE,
    smoothquant_max_scale: float | None = DEFAULT_SMOOTHQUANT_MAX_SCALE,
    smooth_scope: SmoothScope = DEFAULT_SMOOTH_SCOPE,
    smooth_fold: bool = True,
) -> dict[int, BaselineQuantSummary]:
    """Apply SmoothQuant-equivalent FP8 W+A fake quantization to selected layers."""
    layers = get_transformer_layers(model)
    summaries: dict[int, BaselineQuantSummary] = {}
    selected_layer_indices = sorted(layer_indices)
    fp_inputs: list[Batch] | None = None
    stream_layer_idx: int | None = None

    for layer_idx in selected_layer_indices:
        if fp_inputs is None: # 得到fp模型对应layer（外层循环）的input hidden，长度是calib样本数
            fp_inputs = capture_layer_input_batches(
                model=model,
                layer=layers[layer_idx],
                model_batches=model_batches,
            )
            stream_layer_idx = layer_idx
        else:
            if stream_layer_idx is None:
                raise RuntimeError("Internal error: stream_layer_idx is not initialized.")
            while stream_layer_idx < layer_idx:
                fp_inputs = advance_layer_input_batches(
                    layer=layers[stream_layer_idx],
                    batches=fp_inputs,
                )
                stream_layer_idx += 1

        teacher_block = layers[layer_idx]
        scales = collect_smoothquant_scales( # 返回当前层所有linear输入的scale，key是linear在当前层中的名字（如果当前层就是linear，则key是""），value是对应的scale tensor
            teacher_block,
            fp_inputs,
            alpha=smoothquant_alpha,
            min_scale=smoothquant_min_scale,
            max_scale=smoothquant_max_scale,
            smooth_scope=smooth_scope,
        )
        next_fp_inputs = advance_layer_input_batches(layer=teacher_block, batches=fp_inputs)
        quant_block = copy.deepcopy(teacher_block)
        folded_names = (
            fold_smoothquant_scales_inplace(quant_block, scales, smooth_scope=smooth_scope)
            if smooth_fold
            else set()
        )
        quant_block, replaced = smoothquant_quantized_module_from_scales(
            quant_block,
            scales,
            act_quant=act_quant,
            smooth_scope=smooth_scope,
            folded_names=folded_names,
        )
        shared_attention_modules = 0
        shared_mlp_modules = 0
        if act_quant == "per_token" and act_quant_mode == "shared_input":
            shared_attention_modules, shared_mlp_modules = install_shared_input_activation_quantization(quant_block)
        layers[layer_idx] = quant_block
        summaries[layer_idx] = BaselineQuantSummary(
            replaced_linears=replaced,
            skipped_linears=0,
            shared_attention_modules=shared_attention_modules,
            shared_mlp_modules=shared_mlp_modules,
        )
        fp_inputs = next_fp_inputs
        stream_layer_idx = layer_idx + 1
        print(
            f"[smoothquant_w8a8] layer={layer_idx} replaced_linears={replaced}, "
            f"smooth_scope={smooth_scope}, "
            f"smooth_fold={int(smooth_fold)}, folded={len(folded_names)}, "
            f"shared_attention_modules={shared_attention_modules}, "
            f"shared_mlp_modules={shared_mlp_modules}"
        )
    return summaries



def apply_baseline_layers(
    *,
    model: nn.Module,
    layer_indices: Sequence[int],
    act_quant: ActQuant,
    act_quant_mode: ActQuantMode = "per_linear",
) -> dict[int, BaselineQuantSummary]:
    """Apply min-max FP8 W+A fake quantization to selected transformer layers."""
    layers = get_transformer_layers(model)
    summaries: dict[int, BaselineQuantSummary] = {}
    for layer_idx in layer_indices: # 逐层替换nn.Linear为BaselineFakeQuantLinear
        layer = layers[layer_idx]
        if isinstance(layer, nn.Linear):
            layers[layer_idx] = BaselineFakeQuantLinear(layer, act_quant=act_quant)
            summary = BaselineQuantSummary(replaced_linears=1, skipped_linears=0)
        else:
            summary = apply_baseline_w8a8(layer, act_quant=act_quant, act_quant_mode=act_quant_mode)
        summaries[layer_idx] = summary
        print(
            f"[baseline_w8a8] layer={layer_idx} replaced_linears={summary.replaced_linears} "
            f"skipped_linears={summary.skipped_linears}, "
            f"shared_attention_modules={summary.shared_attention_modules}, "
            f"shared_mlp_modules={summary.shared_mlp_modules}"
        )
    return summaries


def apply_learned_quant_params_to_layers(
    model: nn.Module,
    payload: Mapping[str, Any],
    *,
    act_quant_mode: ActQuantMode = "per_linear",
    smooth_scope: SmoothScope = DEFAULT_SMOOTH_SCOPE,
    smooth_fold: bool = True,
) -> list[int]:
    """Apply saved learned quant params to transformer layers and freeze wrappers."""
    layers_payload = payload.get("layers")
    if not isinstance(layers_payload, Mapping):
        raise TypeError("learned quant params payload must contain a layers mapping.")

    layers = get_transformer_layers(model)
    applied: list[int] = []
    for layer_key, layer_params in sorted(layers_payload.items(), key=lambda item: int(item[0])):
        layer_idx = int(layer_key)
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of range for {len(layers)} layers.")
        new_layer, _count = learned_quantized_module_from_params(layers[layer_idx], layer_params)
        if smooth_fold:
            fold_frozen_let_scales_inplace(new_layer, smooth_scope=smooth_scope)
        layers[layer_idx] = new_layer
        if act_quant_mode == "shared_input":
            install_shared_input_activation_quantization(new_layer)
        applied.append(layer_idx)
    return applied


def build_learned_quant_params_payload(
    *,
    method: str,
    act_quant: ActQuant,
    act_quant_mode: ActQuantMode,
    smooth_scope: SmoothScope,
    smooth_fold: bool,
    layers: Mapping[int, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "method": method,
        "quant_format": "fp8_e4m3fn",
        "act_quant": act_quant,
        "act_quant_mode": act_quant_mode,
        "smooth_scope": smooth_scope,
        "smooth_fold": bool(smooth_fold),
        "layers": dict(layers),
    }


def decode_generations(tokenizer: Any, sequences: torch.Tensor, prompt_len: int) -> list[str]:
    generations = []
    for seq in sequences:
        generated_ids = seq[prompt_len:]
        generations.append(tokenizer.decode(generated_ids, skip_special_tokens=False))
    return generations


def generate_one(
    *,
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    input_device: torch.device,
    args: argparse.Namespace,
) -> list[str]:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {
        key: value.to(input_device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    prompt_len = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return decode_generations(tokenizer, output.detach().cpu(), prompt_len)


def result_path(output_dir: str, model_name: str, split: str) -> Path:
    return resolve_repo_path(output_dir) / model_name / "ad" / f"{split}_generated.json"


def save_results(
    *,
    output_file: Path,
    model_name: str,
    split: str,
    test_data: Mapping[str, Mapping[str, Any]],
    generations: Mapping[str, list[str]],
    total_time: float,
    config: Mapping[str, Any],
) -> None:
    samples: dict[str, dict[str, Any]] = {}
    for sample_id, sample in test_data.items():
        item = {
            "prompt": sample.get("prompt", ""),
            "generations": generations.get(sample_id, []),
            "ground_truth": sample.get("ground_truth", ""),
        }
        if "metadata" in sample:
            item["metadata"] = sample["metadata"]
        samples[sample_id] = item

    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "task_name": "ad",
        "split": split,
        "total_time": total_time,
        "avg_time_per_sample": total_time / len(samples) if samples else 0.0,
        "quant_config": dict(config),
        "learnable_quant_config": dict(config),
        "samples": samples,
    }
    output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def maybe_evaluate(output_dir: str, data_dir: str, overwrite: bool) -> None:
    output_root = resolve_repo_path(output_dir)
    data_root = resolve_repo_path(data_dir)
    Benchmark.evaluate_dev(
        generation_results_dir=str(output_root),
        output_path=str(output_root / "eval_results.json"),
        data_dir=str(data_root),
        overwrite=overwrite,
        task_types=["ad"],
    )


def histories_to_jsonable(histories: Mapping[int, CalibrationHistory]) -> dict[str, Any]:
    return {
        str(layer_idx): {
            "initial_loss": history.initial_loss,
            "final_loss": history.final_loss,
            "losses": history.losses,
        }
        for layer_idx, history in histories.items()
    }


def summaries_to_jsonable(summaries: Mapping[int, BaselineQuantSummary]) -> dict[str, Any]:
    return {
        str(layer_idx): {
            "replaced_linears": summary.replaced_linears,
            "skipped_linears": summary.skipped_linears,
            "shared_attention_modules": summary.shared_attention_modules,
            "shared_mlp_modules": summary.shared_mlp_modules,
        }
        for layer_idx, summary in summaries.items()
    }


def _detach_tree(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().clone()
    if isinstance(obj, tuple):
        return tuple(_detach_tree(item) for item in obj)
    if isinstance(obj, list):
        return [_detach_tree(item) for item in obj]
    if isinstance(obj, Mapping):
        return {key: _detach_tree(value) for key, value in obj.items()}
    return obj


def _module_device(module: nn.Module) -> torch.device | None:
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return None


def _move_tree_to_device(obj: Any, device: torch.device | None) -> Any:
    if device is None:
        return obj
    if torch.is_tensor(obj):
        return obj.to(device) if obj.device != device else obj
    if isinstance(obj, tuple):
        return tuple(_move_tree_to_device(item, device) for item in obj)
    if isinstance(obj, list):
        return [_move_tree_to_device(item, device) for item in obj]
    if isinstance(obj, Mapping):
        return {key: _move_tree_to_device(value, device) for key, value in obj.items()}
    return obj


def main() -> None:
    args = parse_args()
    if args.calib_offset < 0 or args.eval_offset < 0:
        raise ValueError("Offsets must be non-negative.")
    set_seed(args.seed)

    # Load model and tokenizer
    model_name = args.model_name or Path(args.model_path.rstrip("/")).name
    output_file = result_path(args.output_dir, model_name, args.split)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and not args.overwrite and not args.calib_only:
        raise FileExistsError(f"Generation file exists: {output_file}. Use --overwrite.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype_from_name(args.dtype),
        "trust_remote_code": True,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    if not args.device_map:
        model = model.to(args.device)
    model.eval()
    input_device = resolve_input_device(model, args.device)

    # Load config and data
    task_config = get_task_config("ad")
    prompt_token = task_config.get("generation_config", {}).get("prompt_token", "<|sid_begin|>")
    calib_data_dir = args.calib_data_dir or args.data_dir
    eval_data_dir = args.eval_data_dir or args.data_dir
    calib_split = args.calib_split or default_calib_split(calib_data_dir, args.split)

    layers = get_transformer_layers(model)
    layer_indices = parse_layer_indices(args.layers, num_layers=len(layers))
    histories: dict[int, CalibrationHistory] = {}
    baseline_summaries: dict[int, BaselineQuantSummary] = {}
    learned_quant_params: dict[int, dict[str, Any]] = {}
    quant_params_payload: dict[str, Any] | None = None
    quant_params_file: Path | None = None

    if args.load_quant_params:
        if args.mode not in {"m1_lwt", "m2_let", "m2_lwt_let"}:
            raise ValueError("--load_quant_params is only supported for learnable modes.")
        load_path = resolve_repo_path(args.load_quant_params)
        try:
            quant_params_payload = torch.load(load_path, map_location="cpu", weights_only=False)
        except TypeError:
            quant_params_payload = torch.load(load_path, map_location="cpu")
        if not isinstance(quant_params_payload, Mapping):
            raise TypeError(f"Expected quant params payload mapping, got {type(quant_params_payload)!r}.")
        payload_method = quant_params_payload.get("method")
        if payload_method is not None and payload_method != args.mode:
            raise ValueError(f"Loaded quant params method={payload_method!r} does not match mode={args.mode!r}.")
        payload_act_quant_mode = quant_params_payload.get("act_quant_mode")
        if payload_act_quant_mode is not None and payload_act_quant_mode != args.act_quant_mode:
            raise ValueError(
                f"Loaded quant params act_quant_mode={payload_act_quant_mode!r} "
                f"does not match requested act_quant_mode={args.act_quant_mode!r}."
            )
        layer_indices = apply_learned_quant_params_to_layers(
            model,
            quant_params_payload,
            act_quant_mode=args.act_quant_mode,
            smooth_scope=args.smooth_scope,
            smooth_fold=args.smooth_fold,
        )
        print(f"[load_quant_params] applied layers={layer_indices} from {load_path}")
    elif args.skip_calibration:
        raise ValueError("--skip_calibration requires --load_quant_params.")
    elif args.mode in {"m1_lwt", "m2_let", "m2_lwt_let"}:
        calib_data = load_ad_data(
            tokenizer,
            str(resolve_repo_path(calib_data_dir)),
            calib_split,
            parse_sample_size(args.calib_sample_size),
            sample_offset=args.calib_offset,
        )
        calib_prompts = [format_prompt(sample["prompt"], prompt_token) for sample in calib_data.values()]
        calib_batches = build_model_batches(
            tokenizer=tokenizer,
            prompts=calib_prompts,
            device=input_device,
        )
        histories = calibrate_model_layers_m1(
            model=model,
            model_batches=calib_batches,
            layer_indices=layer_indices,
            epochs=args.epochs,
            lwt_lr=args.lwt_lr,
            let_lr=args.let_lr,
            act_quant=args.act_quant,
            act_quant_mode=args.act_quant_mode,
            init_clip_multiplier=args.init_clip_multiplier,
            enable_let=args.mode in {"m2_let", "m2_lwt_let"},
            train_lwt=args.mode != "m2_let",
            let_init=args.let_init,
            smoothquant_alpha=args.smoothquant_alpha,
            smoothquant_min_scale=args.smoothquant_min_scale,
            smoothquant_max_scale=args.smoothquant_max_scale,
            smooth_scope=args.smooth_scope,
            smooth_fold=args.smooth_fold,
            learned_quant_params=learned_quant_params,
        )
        quant_params_payload = build_learned_quant_params_payload(
            method=args.mode,
            act_quant=args.act_quant,
            act_quant_mode=args.act_quant_mode,
            smooth_scope=args.smooth_scope,
            smooth_fold=args.smooth_fold,
            layers=learned_quant_params,
        )
    elif args.mode == "baseline_w8a8":
        baseline_summaries = apply_baseline_layers(
            model=model,
            layer_indices=layer_indices,
            act_quant=args.act_quant,
            act_quant_mode=args.act_quant_mode,
        )
    elif args.mode == "smoothquant_w8a8":
        calib_data = load_ad_data(
            tokenizer,
            str(resolve_repo_path(calib_data_dir)),
            calib_split,
            parse_sample_size(args.calib_sample_size),
            sample_offset=args.calib_offset,
        )
        calib_prompts = [format_prompt(sample["prompt"], prompt_token) for sample in calib_data.values()]
        calib_batches = build_model_batches(
            tokenizer=tokenizer,
            prompts=calib_prompts,
            device=input_device,
        )
        baseline_summaries = apply_smoothquant_layers(
            model=model,
            model_batches=calib_batches,
            layer_indices=layer_indices,
            act_quant=args.act_quant,
            act_quant_mode=args.act_quant_mode,
            smoothquant_alpha=args.smoothquant_alpha,
            smoothquant_min_scale=args.smoothquant_min_scale,
            smoothquant_max_scale=args.smoothquant_max_scale,
            smooth_scope=args.smooth_scope,
            smooth_fold=args.smooth_fold,
        )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    if args.save_quant_params and learned_quant_params:
        quant_params_file = output_file.parent / f"{args.mode}_learned_quant_params.pt"
        torch.save(quant_params_payload, quant_params_file)

    config = {
        "method": args.mode,
        "layers": layer_indices,
        "epochs": args.epochs,
        "lwt_lr": args.lwt_lr,
        "let_lr": args.let_lr,
        "let_init": args.let_init,
        "smoothquant_alpha": args.smoothquant_alpha,
        "smooth_scope": args.smooth_scope,
        "smooth_fold": args.smooth_fold,
        "smoothquant_min_scale": args.smoothquant_min_scale,
        "smoothquant_max_scale": args.smoothquant_max_scale,
        "init_clip_multiplier": args.init_clip_multiplier,
        "act_quant": args.act_quant,
        "act_quant_mode": args.act_quant_mode,
        "data_dir": args.data_dir,
        "calib_data_dir": calib_data_dir,
        "eval_data_dir": eval_data_dir,
        "split": args.split,
        "calib_split": calib_split,
        "calib_sample_size": args.calib_sample_size,
        "calib_offset": args.calib_offset,
        "eval_sample_size": args.eval_sample_size,
        "eval_offset": args.eval_offset,
        "dtype": args.dtype,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "save_quant_params": args.save_quant_params,
        "load_quant_params": args.load_quant_params,
        "skip_calibration": args.skip_calibration,
        "quant_params_file": None if quant_params_file is None else str(quant_params_file),
        "histories": histories_to_jsonable(histories),
        "baseline_summaries": summaries_to_jsonable(baseline_summaries),
    }
    config_filename = {
        "m1_lwt": "m1_calibration.json",
        "m2_let": "m2_let_calibration.json",
        "m2_lwt_let": "m2_calibration.json",
        "baseline_w8a8": "baseline_w8a8_config.json",
        "smoothquant_w8a8": "smoothquant_w8a8_config.json",
    }[args.mode]
    (output_file.parent / config_filename).write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.save_model_state:
        torch.save(model.state_dict(), output_file.parent / f"{args.mode}_quantized_model_state.pt")
    if args.calib_only:
        return

    test_data = load_ad_data(
        tokenizer,
        str(resolve_repo_path(eval_data_dir)),
        args.split,
        parse_sample_size(args.eval_sample_size),
        sample_offset=args.eval_offset,
    )
    test_items = list(test_data.items())
    generations: dict[str, list[str]] = {}
    start = time.time()
    for sample_id, sample in tqdm(
        test_items,
        total=len(test_items),
        desc=f"{args.mode} AD generation",
    ):
        prompt = format_prompt(sample["prompt"], prompt_token)
        generations[sample_id] = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            input_device=input_device,
            args=args,
        )
    total_time = time.time() - start

    save_results(
        output_file=output_file,
        model_name=model_name,
        split=args.split,
        test_data=test_data,
        generations=generations,
        total_time=total_time,
        config=config,
    )
    if args.evaluate:
        maybe_evaluate(args.output_dir, eval_data_dir, args.overwrite)


if __name__ == "__main__":
    main()
