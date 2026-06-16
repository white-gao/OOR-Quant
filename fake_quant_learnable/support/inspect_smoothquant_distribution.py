#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..apply import install_shared_input_activation_quantization
from ..modules import BaselineFakeQuantLinear, SmoothQuantFakeQuantLinear
from ..run_m1_onerec_ad import (
    DEFAULT_ACT_QUANT,
    DEFAULT_ACT_QUANT_MODE,
    DEFAULT_DATA_DIR,
    DEFAULT_DTYPE,
    DEFAULT_MODEL_PATH,
    DEFAULT_SMOOTHQUANT_ALPHA,
    DEFAULT_SMOOTHQUANT_MAX_SCALE,
    DEFAULT_SMOOTHQUANT_MIN_SCALE,
    DEFAULT_SMOOTH_FOLD,
    DEFAULT_SMOOTH_SCOPE,
    build_model_batches,
    capture_layer_input_batches,
    default_calib_split,
    dtype_from_name,
    format_prompt,
    get_transformer_layers,
    load_ad_data,
    resolve_repo_path,
)
from .smoothquant_runtime import (
    _batch_to_args_kwargs,
    collect_smoothquant_scales,
    fold_smoothquant_scales_inplace,
    smoothquant_quantized_module_from_scales,
)
from .runtime_utils import _move_tree_to_device
from benchmark.tasks.v1_0.registry import get_task_config
from fake_quant.smoothquant.core import smooth_linear_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect actual baseline vs SmoothQuant+fold activation/weight inputs before FP8 quantization."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--sq_calib_sample_size", type=int, default=3, help="Number of calib samples used to collect SmoothQuant scales.")
    parser.add_argument("--sq_calib_offset", type=int, default=0)
    parser.add_argument("--sample_index", type=int, default=0, help="Single sample offset used for the displayed distributions.")
    parser.add_argument("--layer", default="last", help='Layer index or "last".')
    parser.add_argument("--module", default="mlp.down_proj", help="Linear name inside the selected layer.")
    parser.add_argument("--token_index", type=int, default=-1, help="Token position to inspect for activation stats.")
    parser.add_argument("--smoothquant_alpha", type=float, default=DEFAULT_SMOOTHQUANT_ALPHA)
    parser.add_argument("--output", default="fake_quant_learnable/results/smoothquant_distribution_last_layer.json")
    parser.add_argument("--plot", default=None, help="Optional histogram png path.")
    return parser.parse_args()


def get_submodule(module: nn.Module, name: str) -> nn.Module:
    try:
        return module.get_submodule(name)
    except AttributeError as exc:
        raise ValueError(f"Module {name!r} not found.") from exc


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    x = tensor.detach().float().cpu().reshape(-1)
    if x.numel() == 0:
        raise ValueError("Cannot summarize an empty tensor.")
    abs_x = x.abs()
    quantiles = torch.quantile(abs_x, torch.tensor([0.5, 0.9, 0.95, 0.99, 0.999]))
    return {
        "shape": list(tensor.shape),
        "numel": int(x.numel()),
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "min": float(x.min()),
        "max": float(x.max()),
        "abs_mean": float(abs_x.mean()),
        "abs_max": float(abs_x.max()),
        "abs_p50": float(quantiles[0]),
        "abs_p90": float(quantiles[1]),
        "abs_p95": float(quantiles[2]),
        "abs_p99": float(quantiles[3]),
        "abs_p999": float(quantiles[4]),
    }


def channel_absmax_stats(tensor: torch.Tensor, dim: int) -> dict[str, Any]:
    x = tensor.detach().float().abs()
    reduce_dims = tuple(i for i in range(x.ndim) if i != dim)
    per_channel = x.amax(dim=reduce_dims) if reduce_dims else x
    return tensor_stats(per_channel)


def first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    if isinstance(output, Mapping):
        for item in output.values():
            if torch.is_tensor(item):
                return item
    raise TypeError(f"No tensor found in output type {type(output)!r}.")


def capture_module_input(layer: nn.Module, target_name: str, layer_batch: Any) -> torch.Tensor:
    target = get_submodule(layer, target_name)
    captured: list[torch.Tensor] = []

    def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if args:
            x = args[0]
        else:
            x = kwargs.get("input", kwargs.get("hidden_states"))
        if not torch.is_tensor(x):
            raise TypeError(f"Captured input for {target_name!r} is not a tensor.")
        captured.append(x.detach().clone())

    handle = target.register_forward_pre_hook(hook, with_kwargs=True)
    was_training = layer.training
    layer.eval()
    try:
        with torch.no_grad():
            args, kwargs = _batch_to_args_kwargs(layer_batch)
            device = next(layer.parameters()).device
            args = _move_tree_to_device(args, device)
            kwargs = _move_tree_to_device(kwargs, device)
            layer(*args, **kwargs)
    finally:
        handle.remove()
        layer.train(was_training)
    if not captured:
        raise RuntimeError(f"No input captured for {target_name!r}.")
    return captured[0]


def activation_quant_input(wrapper: nn.Module, wrapper_input: torch.Tensor) -> torch.Tensor:
    if isinstance(wrapper, SmoothQuantFakeQuantLinear):
        return wrapper.smooth_activation(wrapper_input)
    if isinstance(wrapper, BaselineFakeQuantLinear):
        return wrapper_input
    raise TypeError(f"Expected quant Linear wrapper, got {type(wrapper)!r}.")


def baseline_quant_block(block: nn.Module, *, shared_input: bool = False) -> nn.Module:
    quant_block = copy.deepcopy(block)
    for _parent_name, parent in list(quant_block.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                setattr(parent, child_name, BaselineFakeQuantLinear(child, act_quant=DEFAULT_ACT_QUANT))
    if shared_input and DEFAULT_ACT_QUANT == "per_token" and DEFAULT_ACT_QUANT_MODE == "shared_input":
        install_shared_input_activation_quantization(quant_block)
    return quant_block


def smoothquant_block_and_weight_input(
    block: nn.Module,
    layer_inputs: list[Any],
    module_name: str,
    alpha: float,
    shared_input: bool = False,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor, set[str]]:
    scales = collect_smoothquant_scales(
        block,
        layer_inputs,
        alpha=alpha,
        min_scale=DEFAULT_SMOOTHQUANT_MIN_SCALE,
        max_scale=DEFAULT_SMOOTHQUANT_MAX_SCALE,
        smooth_scope=DEFAULT_SMOOTH_SCOPE,
    )
    if module_name not in scales:
        raise KeyError(f"No SmoothQuant scale found for {module_name!r}. Available: {sorted(scales)}")

    quant_block = copy.deepcopy(block)
    folded_names = (
        fold_smoothquant_scales_inplace(quant_block, scales, smooth_scope=DEFAULT_SMOOTH_SCOPE)
        if DEFAULT_SMOOTH_FOLD
        else set()
    )
    folded_linear = get_submodule(quant_block, module_name)
    if not isinstance(folded_linear, nn.Linear):
        raise TypeError(f"Expected folded module {module_name!r} to still be nn.Linear before wrapping.")
    scale = scales[module_name].to(device=folded_linear.weight.device)
    if module_name in folded_names:
        weight_quant_input = folded_linear.weight.detach().clone()
    else:
        weight_quant_input = smooth_linear_weight(folded_linear.weight.detach(), scale).detach().clone()

    quant_block, _replaced = smoothquant_quantized_module_from_scales(
        quant_block,
        scales,
        act_quant=DEFAULT_ACT_QUANT,
        smooth_scope=DEFAULT_SMOOTH_SCOPE,
        folded_names=folded_names,
    )
    if shared_input and DEFAULT_ACT_QUANT == "per_token" and DEFAULT_ACT_QUANT_MODE == "shared_input":
        install_shared_input_activation_quantization(quant_block)
    return quant_block, weight_quant_input, scales[module_name].detach().clone(), folded_names



def run_layer_output(layer: nn.Module, layer_batch: Any) -> torch.Tensor:
    was_training = layer.training
    layer.eval()
    try:
        with torch.no_grad():
            args, kwargs = _batch_to_args_kwargs(layer_batch)
            device = next(layer.parameters()).device
            args = _move_tree_to_device(args, device)
            kwargs = _move_tree_to_device(kwargs, device)
            return first_tensor(layer(*args, **kwargs)).detach().float().cpu()
    finally:
        layer.train(was_training)


def mse_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    diff = candidate.float() - reference.float()
    per_token = diff.pow(2).mean(dim=-1) if diff.ndim >= 2 else diff.pow(2)
    return {
        "mse": float(diff.pow(2).mean()),
        "rmse": float(diff.pow(2).mean().sqrt()),
        "mae": float(diff.abs().mean()),
        "max_abs_error": float(diff.abs().max()),
        "per_token_mse": tensor_stats(per_token),
        "output": tensor_stats(candidate),
        "error": tensor_stats(diff),
    }

def select_token(x: torch.Tensor, token_index: int) -> torch.Tensor:
    if x.ndim < 2:
        return x
    if x.ndim >= 3:
        return x[:, token_index, :]
    return x[token_index, :].unsqueeze(0)


def plot_histograms(path: str, baseline_act: torch.Tensor, smooth_act: torch.Tensor, baseline_weight: torch.Tensor, smooth_weight: torch.Tensor) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    pairs = [
        (axes[0, 0], baseline_act, "baseline activation quant input"),
        (axes[0, 1], smooth_act, "smoothquant+fold activation quant input"),
        (axes[1, 0], baseline_weight, "baseline weight quant input"),
        (axes[1, 1], smooth_weight, "smoothquant+fold weight quant input"),
    ]
    for ax, tensor, title in pairs:
        values = tensor.detach().float().cpu().reshape(-1).numpy()
        ax.hist(values, bins=120)
        ax.set_title(title)
        ax.set_yscale("log")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_from_name(args.dtype),
        trust_remote_code=True,
    ).to(device)
    model.eval()

    data_dir = str(resolve_repo_path(args.data_dir))
    split = default_calib_split(args.data_dir, "test")
    task_config = get_task_config("ad")
    prompt_token = task_config.get("generation_config", {}).get("prompt_token", "<|sid_begin|>")

    calib_data = load_ad_data(
        tokenizer,
        data_dir,
        split,
        args.sq_calib_sample_size,
        sample_offset=args.sq_calib_offset,
    )
    calib_prompts = [format_prompt(sample["prompt"], prompt_token) for sample in calib_data.values()]
    calib_batches = build_model_batches(tokenizer=tokenizer, prompts=calib_prompts, device=device)

    inspect_data = load_ad_data(tokenizer, data_dir, split, 1, sample_offset=args.sample_index)
    inspect_sample = next(iter(inspect_data.values()))
    inspect_prompt = format_prompt(inspect_sample["prompt"], prompt_token)
    inspect_batches = build_model_batches(tokenizer=tokenizer, prompts=[inspect_prompt], device=device)

    layers = get_transformer_layers(model)
    layer_idx = len(layers) - 1 if args.layer == "last" else int(args.layer)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"layer index {layer_idx} out of range for {len(layers)} layers.")

    sq_layer_inputs = capture_layer_input_batches(model=model, layer=layers[layer_idx], model_batches=calib_batches)
    inspect_layer_inputs = capture_layer_input_batches(model=model, layer=layers[layer_idx], model_batches=inspect_batches)
    layer_batch = inspect_layer_inputs[0]
    teacher_block = layers[layer_idx]
    baseline_linear = get_submodule(teacher_block, args.module)
    if not isinstance(baseline_linear, nn.Linear):
        raise TypeError(f"{args.module!r} is not an nn.Linear in the teacher block.")

    baseline_block = baseline_quant_block(teacher_block)
    baseline_wrapper = get_submodule(baseline_block, args.module)
    baseline_wrapper_input = capture_module_input(baseline_block, args.module, layer_batch)
    baseline_act_input = activation_quant_input(baseline_wrapper, baseline_wrapper_input)
    baseline_weight_input = baseline_linear.weight.detach().clone()

    smooth_block, smooth_weight_input, smooth_scale, folded_names = smoothquant_block_and_weight_input(
        teacher_block,
        sq_layer_inputs,
        args.module,
        args.smoothquant_alpha,
    )
    smooth_wrapper = get_submodule(smooth_block, args.module)
    smooth_wrapper_input = capture_module_input(smooth_block, args.module, layer_batch)
    smooth_act_input = activation_quant_input(smooth_wrapper, smooth_wrapper_input)

    teacher_output = run_layer_output(teacher_block, layer_batch)
    baseline_mse_block = baseline_quant_block(teacher_block, shared_input=True)
    smooth_mse_block, _smooth_mse_weight_input, _smooth_mse_scale, _smooth_mse_folded = smoothquant_block_and_weight_input(
        teacher_block,
        sq_layer_inputs,
        args.module,
        args.smoothquant_alpha,
        shared_input=True,
    )
    baseline_output = run_layer_output(baseline_mse_block, layer_batch)
    smooth_output = run_layer_output(smooth_mse_block, layer_batch)

    baseline_act_token = select_token(baseline_act_input, args.token_index)
    smooth_act_token = select_token(smooth_act_input, args.token_index)
    result = {
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "split": split,
        "sq_calib_sample_size": args.sq_calib_sample_size,
        "sq_calib_offset": args.sq_calib_offset,
        "sample_index": args.sample_index,
        "layer_index": layer_idx,
        "module": args.module,
        "token_index": args.token_index,
        "smoothquant_alpha": args.smoothquant_alpha,
        "smooth_scope": DEFAULT_SMOOTH_SCOPE,
        "smooth_fold": DEFAULT_SMOOTH_FOLD,
        "capture_note": "shared_input monkeypatch is not installed in this inspection script so module hooks can capture activation inputs; the tensor before activation QDQ is unchanged for the inspected Linear input.",
        "module_folded": args.module in folded_names,
        "folded_names": sorted(folded_names),
        "scale": tensor_stats(smooth_scale),
        "last_layer_output_mse": {
            "teacher": tensor_stats(teacher_output),
            "baseline_w8a8": mse_stats(teacher_output, baseline_output),
            "smoothquant_w8a8_fold": mse_stats(teacher_output, smooth_output),
            "smooth_over_baseline_mse_ratio": float(
                (smooth_output.float() - teacher_output.float()).pow(2).mean()
                / ((baseline_output.float() - teacher_output.float()).pow(2).mean() + 1e-12)
            ),
        },
        "activation_quant_input": {
            "baseline_all_tokens": tensor_stats(baseline_act_input),
            "smoothquant_fold_all_tokens": tensor_stats(smooth_act_input),
            "baseline_selected_token": tensor_stats(baseline_act_token),
            "smoothquant_fold_selected_token": tensor_stats(smooth_act_token),
            "baseline_channel_absmax": channel_absmax_stats(baseline_act_input, dim=baseline_act_input.ndim - 1),
            "smoothquant_fold_channel_absmax": channel_absmax_stats(smooth_act_input, dim=smooth_act_input.ndim - 1),
        },
        "weight_quant_input": {
            "baseline": tensor_stats(baseline_weight_input),
            "smoothquant_fold": tensor_stats(smooth_weight_input),
            "baseline_input_channel_absmax": channel_absmax_stats(baseline_weight_input, dim=1),
            "smoothquant_fold_input_channel_absmax": channel_absmax_stats(smooth_weight_input, dim=1),
            "baseline_output_channel_absmax": channel_absmax_stats(baseline_weight_input, dim=0),
            "smoothquant_fold_output_channel_absmax": channel_absmax_stats(smooth_weight_input, dim=0),
        },
    }

    output_path = resolve_repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved stats to {output_path}")
    if args.plot:
        plot_histograms(args.plot, baseline_act_token, smooth_act_token, baseline_weight_input, smooth_weight_input)
        print(f"Saved histogram to {resolve_repo_path(args.plot)}")


if __name__ == "__main__":
    main()
