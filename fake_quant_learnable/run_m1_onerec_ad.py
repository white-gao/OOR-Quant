#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from .modules import BaselineFakeQuantLinear, LearnableFakeQuantLinear
from .quant import ActQuant, ActQuantMode
from .runtime_utils import _detach_tree, _module_device, _move_tree_to_device
from .smoothquant_runtime import (
    DEFAULT_SMOOTHQUANT_ALPHA,
    DEFAULT_SMOOTHQUANT_MAX_SCALE,
    DEFAULT_SMOOTHQUANT_MIN_SCALE,
    DEFAULT_SMOOTH_FOLD,
    DEFAULT_SMOOTH_SCOPE,
    apply_smoothquant_scales_to_learnable,
    collect_smoothquant_scales,
    fold_frozen_let_scales_inplace,
    fold_smoothquant_scales_inplace,
    smoothquant_quantized_module_from_scales,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
for path in (PROJECT_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark import Benchmark  # noqa: E402
from benchmark.tasks.v1_0.registry import get_loader, get_task_config  # noqa: E402


DEFAULT_MODEL_PATH = "/home/guowei/OneRec-1.7B/"
DEFAULT_DATA_DIR = "data/onerec_data/benchmark-data-calib1024"
DEFAULT_OUTPUT_DIR = "fake_quant_learnable/results/ptq_ad"
DEFAULT_SPLIT = "test"
DEFAULT_CALIB_OFFSET = 0
DEFAULT_EVAL_OFFSET = 0
DEFAULT_ACT_QUANT: ActQuant = "per_token"
DEFAULT_ACT_QUANT_MODE: ActQuantMode = "shared_input"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_NUM_BEAMS = 32
DEFAULT_NUM_RETURN_SEQUENCES = 32
DEFAULT_MAX_NEW_TOKENS = 3
DEFAULT_SEED = 42
DEFAULT_INIT_CLIP_MULTIPLIER = 1.0
DEFAULT_SID_PPL_MAX_ITEMS = 1
SID_ITEM_RE = re.compile(
    r"<\|sid_begin\|>"
    r"(?P<sid><s_a_[^>]+><s_b_[^>]+><s_c_[^>]+>)"
    r"<\|sid_end\|>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OneRec AD evaluation with compact FP8 W+A quantization defaults."
    )
    parser.add_argument("--mode", default="m2_lwt_let", choices=["baseline_w8a8", "smoothquant_w8a8", "m1_lwt", "m2_let", "m2_lwt_let"])
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", default="all", help='Layer spec: "all", "last:K", "0,2-4".')
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calib_sample_size", default="1024")
    parser.add_argument("--eval_sample_size", default="full")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lwt_lr", type=float, default=DEFAULT_LWT_LR)
    parser.add_argument("--let_lr", type=float, default=DEFAULT_LET_LR)
    parser.add_argument("--let_init", default="ones", choices=["ones", "smoothquant"])
    parser.add_argument("--load_quant_params", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--calib_only", action="store_true")
    parser.add_argument(
        "--compute_sid_ppl",
        action="store_true",
        help="Compute auxiliary teacher-forcing NLL/PPL on ground-truth SID tokens.",
    )
    parser.add_argument(
        "--sid_ppl_max_items",
        type=int,
        default=DEFAULT_SID_PPL_MAX_ITEMS,
        help="Maximum ground-truth SID items per sample for teacher-forcing NLL/PPL.",
    )
    args = parser.parse_args()
    _attach_fixed_defaults(args)
    return args


def _attach_fixed_defaults(args: argparse.Namespace) -> None:
    args.split = DEFAULT_SPLIT
    args.calib_offset = DEFAULT_CALIB_OFFSET
    args.eval_offset = DEFAULT_EVAL_OFFSET
    args.act_quant = DEFAULT_ACT_QUANT
    args.act_quant_mode = DEFAULT_ACT_QUANT_MODE
    args.dtype = DEFAULT_DTYPE
    args.num_beams = DEFAULT_NUM_BEAMS
    args.num_return_sequences = DEFAULT_NUM_RETURN_SEQUENCES
    args.max_new_tokens = DEFAULT_MAX_NEW_TOKENS
    args.seed = DEFAULT_SEED
    args.init_clip_multiplier = DEFAULT_INIT_CLIP_MULTIPLIER
    args.smoothquant_alpha = DEFAULT_SMOOTHQUANT_ALPHA
    args.smooth_scope = DEFAULT_SMOOTH_SCOPE
    args.smooth_fold = DEFAULT_SMOOTH_FOLD
    args.smoothquant_min_scale = DEFAULT_SMOOTHQUANT_MIN_SCALE
    args.smoothquant_max_scale = DEFAULT_SMOOTHQUANT_MAX_SCALE


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


def extract_sid_teacher_forcing_targets(ground_truth: str, *, max_items: int) -> list[str]:
    """Return SID target triples without the surrounding sid_begin/sid_end tokens."""
    if max_items <= 0:
        return []
    targets: list[str] = []
    for match in SID_ITEM_RE.finditer(ground_truth or ""):
        targets.append(match.group("sid"))
        if len(targets) >= max_items:
            break
    return targets


def _safe_exp(value: float) -> float:
    return float(math.exp(min(value, 50.0)))


def compute_sid_teacher_forcing_metrics(
    *,
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    ground_truth: str,
    input_device: torch.device,
    max_items: int = DEFAULT_SID_PPL_MAX_ITEMS,
) -> dict[str, Any]:
    """Compute teacher-forcing NLL/PPL for ground-truth SID tokens.

    The AD prompt already ends with <|sid_begin|>. This metric scores the next
    ground-truth <s_a_*><s_b_*><s_c_*> tokens, not sid_begin/sid_end.
    """
    targets = extract_sid_teacher_forcing_targets(ground_truth, max_items=max_items)
    if not targets:
        return {
            "sid_tf_valid": False,
            "sid_tf_num_items": 0,
            "sid_tf_num_tokens": 0,
        }

    prompt_encoded = tokenizer(prompt, return_tensors="pt")
    prompt_input_ids = prompt_encoded["input_ids"]
    prompt_len = int(prompt_input_ids.shape[-1])
    prompt_attention_mask = prompt_encoded.get("attention_mask")

    total_loss = 0.0
    total_tokens = 0
    valid_items = 0
    first_target = targets[0]

    for target_text in targets:
        target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        target_len = int(target_ids.shape[-1])
        if target_len == 0:
            continue

        input_ids = torch.cat([prompt_input_ids, target_ids], dim=-1).to(input_device)
        model_inputs: dict[str, torch.Tensor] = {"input_ids": input_ids}
        if prompt_attention_mask is not None:
            target_mask = torch.ones_like(target_ids)
            attention_mask = torch.cat([prompt_attention_mask, target_mask], dim=-1).to(input_device)
            model_inputs["attention_mask"] = attention_mask

        labels = target_ids.reshape(-1).to(input_device)
        positions = torch.arange(
            prompt_len - 1,
            prompt_len - 1 + target_len,
            device=input_device,
        )
        with torch.inference_mode():
            try:
                outputs = model(**model_inputs, use_cache=False)
            except TypeError:
                outputs = model(**model_inputs)
        logits = outputs.logits[0, positions, :].float()
        losses = F.cross_entropy(logits, labels, reduction="none")
        total_loss += float(losses.sum().item())
        total_tokens += target_len
        valid_items += 1

    if total_tokens == 0:
        return {
            "sid_tf_valid": False,
            "sid_tf_num_items": 0,
            "sid_tf_num_tokens": 0,
        }

    mean_nll = total_loss / total_tokens
    return {
        "sid_tf_valid": True,
        "sid_tf_nll": mean_nll,
        "sid_tf_ppl": _safe_exp(mean_nll),
        "sid_tf_num_items": valid_items,
        "sid_tf_num_tokens": total_tokens,
        "sid_tf_target": first_target,
    }


def aggregate_sid_teacher_forcing_metrics(
    sample_metrics: Mapping[str, Mapping[str, Any]],
    *,
    max_items: int,
) -> dict[str, Any]:
    total_samples = len(sample_metrics)
    valid_samples = 0
    total_tokens = 0
    weighted_nll = 0.0
    total_items = 0
    for metrics in sample_metrics.values():
        if not metrics.get("sid_tf_valid"):
            continue
        num_tokens = int(metrics.get("sid_tf_num_tokens", 0))
        if num_tokens <= 0:
            continue
        valid_samples += 1
        total_items += int(metrics.get("sid_tf_num_items", 0))
        total_tokens += num_tokens
        weighted_nll += float(metrics["sid_tf_nll"]) * num_tokens

    mean_nll = weighted_nll / total_tokens if total_tokens else None
    return {
        "sid_tf_enabled": True,
        "sid_tf_definition": "teacher_forcing_gt_sid_tokens_excluding_sid_begin_end",
        "sid_tf_max_items_per_sample": max_items,
        "sid_tf_total_samples": total_samples,
        "sid_tf_valid_samples": valid_samples,
        "sid_tf_invalid_samples": total_samples - valid_samples,
        "sid_tf_num_items": total_items,
        "sid_tf_num_tokens": total_tokens,
        "sid_tf_nll": mean_nll,
        "sid_tf_ppl": None if mean_nll is None else _safe_exp(mean_nll),
    }



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
    sample_aux_metrics: Mapping[str, Mapping[str, Any]] | None = None,
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
        if sample_aux_metrics and sample_id in sample_aux_metrics:
            item.update(sample_aux_metrics[sample_id])
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


def merge_sid_teacher_forcing_metrics_into_eval(
    *,
    output_dir: str,
    model_name: str,
    split: str,
    metrics: Mapping[str, Any],
) -> None:
    eval_path = resolve_repo_path(output_dir) / "eval_results.json"
    if not eval_path.exists():
        return
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    model_metrics = data.setdefault(model_name, {})
    task_metrics = model_metrics.setdefault("ad", {})
    split_metrics = task_metrics.setdefault(split, {})
    split_metrics.update(dict(metrics))
    eval_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")



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


def main() -> None:
    args = parse_args()
    if args.sid_ppl_max_items <= 0:
        raise ValueError(f"--sid_ppl_max_items must be positive, got {args.sid_ppl_max_items}")
    set_seed(args.seed)

    # Load model and tokenizer
    model_name = Path(args.model_path.rstrip("/")).name
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
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model = model.to(args.device)
    model.eval()
    input_device = resolve_input_device(model, args.device)

    # Load config and data
    task_config = get_task_config("ad")
    prompt_token = task_config.get("generation_config", {}).get("prompt_token", "<|sid_begin|>")
    calib_data_dir = args.data_dir
    eval_data_dir = args.data_dir
    calib_split = default_calib_split(calib_data_dir, args.split)

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

    if learned_quant_params:
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
        "compute_sid_ppl": args.compute_sid_ppl,
        "sid_ppl_max_items": args.sid_ppl_max_items,
        "seed": args.seed,
        "save_quant_params": True,
        "load_quant_params": args.load_quant_params,
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
    sample_aux_metrics: dict[str, dict[str, Any]] = {}
    sid_tf_total_time = 0.0
    start = time.time()
    for sample_id, sample in tqdm(
        test_items,
        total=len(test_items),
        desc=f"{args.mode} AD generation",
    ):
        prompt = format_prompt(sample["prompt"], prompt_token)
        if args.compute_sid_ppl:
            sid_tf_start = time.time()
            sample_aux_metrics[sample_id] = compute_sid_teacher_forcing_metrics(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                ground_truth=sample.get("ground_truth", ""),
                input_device=input_device,
                max_items=args.sid_ppl_max_items,
            )
            sid_tf_total_time += time.time() - sid_tf_start
        generations[sample_id] = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            input_device=input_device,
            args=args,
        )
    raw_total_time = time.time() - start
    total_time = raw_total_time - sid_tf_total_time if args.compute_sid_ppl else raw_total_time

    sid_tf_metrics: dict[str, Any] = {}
    if args.compute_sid_ppl:
        sid_tf_metrics = aggregate_sid_teacher_forcing_metrics(
            sample_aux_metrics,
            max_items=args.sid_ppl_max_items,
        )
        sid_tf_metrics["sid_tf_total_time"] = sid_tf_total_time
        sid_tf_metrics["sid_tf_avg_time_per_sample"] = sid_tf_total_time / len(test_items) if test_items else 0.0
        config["sid_teacher_forcing_metrics"] = sid_tf_metrics
        (output_file.parent / config_filename).write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    save_results(
        output_file=output_file,
        model_name=model_name,
        split=args.split,
        test_data=test_data,
        generations=generations,
        total_time=total_time,
        config=config,
        sample_aux_metrics=sample_aux_metrics if args.compute_sid_ppl else None,
    )
    if args.evaluate:
        maybe_evaluate(args.output_dir, eval_data_dir, args.overwrite)
        if sid_tf_metrics:
            merge_sid_teacher_forcing_metrics_into_eval(
                output_dir=args.output_dir,
                model_name=model_name,
                split=args.split,
                metrics=sid_tf_metrics,
            )


if __name__ == "__main__":
    main()
