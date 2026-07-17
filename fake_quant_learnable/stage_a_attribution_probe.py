"""Stage-A weight/activation attribution on a fake GPTQ W8A8 model.

The real-runtime stage probe established that activation rescue at Stage A is
useful. This companion probe retains BF16 weight snapshots solely to identify
whether that gain is activation-only or coupled with weight quantization.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from fake_quant_learnable import apply as fake_apply
from fake_quant_learnable.gptq import (
    DEFAULT_GPTQ_BLOCK_SIZE,
    DEFAULT_GPTQ_DAMP_PERCENT,
    collect_gptq_hessians,
    gptq_quantized_module_from_hessians,
)
from fake_quant_learnable.modules import GPTQFakeQuantLinear
from fake_quant_learnable.quant import activation_per_token_qdq_forward_tail_protected
from fake_quant_learnable.run_m1_onerec_ad import (
    BaselineQuantSummary,
    advance_layer_input_batches,
    build_model_batches,
    capture_layer_input_batches,
    default_calib_split,
    format_prompt,
    generate_one,
    get_transformer_layers,
    load_task_data,
    maybe_evaluate,
    parse_layer_indices,
    parse_sample_size,
    resolve_input_device,
    resolve_repo_path,
    result_path,
    save_results,
)
from real_quant.naive_w8a8.run_sid_stage_probe import paired_recovery
from real_quant.naive_w8a8.stage_rescue import current_probe_stage, install_stage_rescue_model_hook, stage_rescue_context


DEFAULT_OUTPUT_DIR = "fake_quant_learnable/results/analysis/sid_stage_a_attribution"
MODES = ("w8a8", "a16", "w16", "wa16")
_ATTRIBUTION_MODE: ContextVar[str | None] = ContextVar("sid_stage_a_attribution_mode", default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-A W8A8/A16/W16/WA16 fake-GPTQ attribution probe.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", default="data/onerec_data/benchmark-data-calib1024")
    parser.add_argument("--task", default="ad")
    parser.add_argument("--split", default="test")
    parser.add_argument("--calib_split", default="calib")
    parser.add_argument("--calib_sample_size", default="128")
    parser.add_argument("--eval_sample_size", default="1000")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num_beams", type=int, default=32)
    parser.add_argument("--num_return_sequences", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    invalid = set(modes) - set(MODES)
    if invalid:
        raise ValueError(f"Unsupported attribution modes: {sorted(invalid)}")
    if "w8a8" not in modes:
        raise ValueError("The attribution probe requires w8a8 as the paired baseline.")
    return modes


def _snapshot_linear_weights(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: child.weight.detach().clone()
        for name, child in module.named_modules()
        if isinstance(child, nn.Linear)
    }


def _attach_fp_weight_snapshots(module: nn.Module, snapshots: Mapping[str, torch.Tensor]) -> None:
    for name, child in module.named_modules():
        if not isinstance(child, GPTQFakeQuantLinear):
            continue
        if name not in snapshots:
            raise KeyError(f"Missing FP16/BF16 snapshot for quantized Linear {name!r}")
        child.register_buffer("stage_probe_weight_fp", snapshots[name].detach().clone(), persistent=False)


def apply_gptq_with_fp_snapshots(
    *,
    model: nn.Module,
    model_batches: Sequence[Mapping[str, Any]],
    layer_indices: Sequence[int],
    damp_percent: float,
    block_size: int,
) -> dict[int, BaselineQuantSummary]:
    """GPTQ calibration equivalent to the standard fake runner plus BF16 snapshots."""
    layers = get_transformer_layers(model)
    summaries: dict[int, BaselineQuantSummary] = {}
    fp_inputs = None
    stream_layer_idx: int | None = None
    for layer_idx in sorted(layer_indices):
        if fp_inputs is None:
            fp_inputs = capture_layer_input_batches(model=model, layer=layers[layer_idx], model_batches=model_batches)
            stream_layer_idx = layer_idx
        else:
            assert stream_layer_idx is not None
            while stream_layer_idx < layer_idx:
                fp_inputs = advance_layer_input_batches(layer=layers[stream_layer_idx], batches=fp_inputs)
                stream_layer_idx += 1

        teacher_block = layers[layer_idx]
        snapshots = _snapshot_linear_weights(teacher_block)
        hessians = collect_gptq_hessians(teacher_block, fp_inputs)
        next_fp_inputs = advance_layer_input_batches(layer=teacher_block, batches=fp_inputs)
        quant_block = copy.deepcopy(teacher_block)
        quant_block, replaced = gptq_quantized_module_from_hessians(
            quant_block,
            hessians,
            act_quant="per_token",
            damp_percent=damp_percent,
            block_size=block_size,
        )
        _attach_fp_weight_snapshots(quant_block, snapshots)
        shared_attention, shared_mlp = fake_apply.install_shared_input_activation_quantization(quant_block)
        layers[layer_idx] = quant_block
        summaries[layer_idx] = BaselineQuantSummary(
            replaced_linears=replaced,
            skipped_linears=0,
            shared_attention_modules=shared_attention,
            shared_mlp_modules=shared_mlp,
        )
        fp_inputs = next_fp_inputs
        stream_layer_idx = layer_idx + 1
        print(f"[stage_a_attribution] gptq layer={layer_idx} linears={replaced}")
    return summaries


def _stage_a_active() -> bool:
    return current_probe_stage() == "a" and _ATTRIBUTION_MODE.get() in {"a16", "w16", "wa16"}


@contextlib.contextmanager
def stage_a_attribution_context(model: nn.Module, mode: str) -> Iterator[None]:
    """Apply one Stage-A attribution path without changing the base model."""
    if mode not in MODES:
        raise ValueError(f"Unsupported attribution mode {mode!r}")
    if mode == "w8a8":
        yield
        return

    original_prepare = fake_apply._shared_prepare_input
    original_forward_prepared = GPTQFakeQuantLinear.forward_prepared

    def stage_prepare(modules: tuple[Any, ...], x: torch.Tensor) -> torch.Tensor | None:
        if _stage_a_active() and _ATTRIBUTION_MODE.get() in {"a16", "wa16"}:
            quant_modules = [module for module in modules if isinstance(module, GPTQFakeQuantLinear)]
            if quant_modules and any(module.act_quant == "per_token" for module in quant_modules):
                first = quant_modules[0]
                return activation_per_token_qdq_forward_tail_protected(
                    x,
                    tail_tokens=1,
                    qmax=first.qmax,
                    eps=first.eps,
                )
        return original_prepare(modules, x)

    def stage_forward_prepared(module: GPTQFakeQuantLinear, x: torch.Tensor) -> torch.Tensor:
        if _stage_a_active() and _ATTRIBUTION_MODE.get() in {"w16", "wa16"} and x.ndim >= 3:
            weight_fp = getattr(module, "stage_probe_weight_fp", None)
            if weight_fp is None:
                raise RuntimeError("Missing stage_probe_weight_fp on GPTQ fake-quant Linear.")
            x_main = x[..., :-1, :]
            x_tail = x[..., -1:, :]
            y_main = F.linear(x_main, module.weight_qdq, module.bias)
            y_tail = F.linear(x_tail, weight_fp.to(dtype=x_tail.dtype), module.bias)
            return torch.cat([y_main, y_tail], dim=-2)
        return original_forward_prepared(module, x)

    hook = install_stage_rescue_model_hook(model)
    mode_token = _ATTRIBUTION_MODE.set(mode)
    fake_apply._shared_prepare_input = stage_prepare
    GPTQFakeQuantLinear.forward_prepared = stage_forward_prepared  # type: ignore[method-assign]
    try:
        with stage_rescue_context({"a"}):
            yield
    finally:
        GPTQFakeQuantLinear.forward_prepared = original_forward_prepared  # type: ignore[method-assign]
        fake_apply._shared_prepare_input = original_prepare
        _ATTRIBUTION_MODE.reset(mode_token)
        hook.remove()


def metrics_block(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    model = next(value for key, value in raw.items() if key != "_total_time")
    task = next(value for key, value in model.items() if key != "_total_time")
    return next(value for key, value in task.items() if key != "_total_time")


def main() -> None:
    args = parse_args()
    modes = parse_modes(args.modes)
    torch_dtype = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16}.get(args.dtype)
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype {args.dtype!r}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch_dtype).to(args.device)
    model.eval()
    input_device = resolve_input_device(model, args.device)

    from benchmark.tasks.v1_0.registry import get_task_config

    task_config = get_task_config(args.task)
    prompt_token = task_config.get("generation_config", {}).get("prompt_token", "<|sid_begin|>")
    calib_data = load_task_data(
        task_name=args.task,
        tokenizer=tokenizer,
        data_dir=str(resolve_repo_path(args.data_dir)),
        split=args.calib_split or default_calib_split(args.data_dir, args.split, task_name=args.task),
        sample_size=parse_sample_size(args.calib_sample_size),
        sample_offset=0,
    )
    calib_prompts = [format_prompt(sample["prompt"], prompt_token) for sample in calib_data.values()]
    calib_batches = build_model_batches(tokenizer=tokenizer, prompts=calib_prompts, device=input_device)
    layers = get_transformer_layers(model)
    summaries = apply_gptq_with_fp_snapshots(
        model=model,
        model_batches=calib_batches,
        layer_indices=parse_layer_indices("all", num_layers=len(layers)),
        damp_percent=DEFAULT_GPTQ_DAMP_PERCENT,
        block_size=DEFAULT_GPTQ_BLOCK_SIZE,
    )

    test_data = load_task_data(
        task_name=args.task,
        tokenizer=tokenizer,
        data_dir=str(resolve_repo_path(args.data_dir)),
        split=args.split,
        sample_size=parse_sample_size(args.eval_sample_size),
        sample_offset=0,
    )
    output_root = resolve_repo_path(args.output_dir)
    common_config = {
        "probe": "stage_a_weight_activation_attribution",
        "base_quantization": "fake GPTQ FP8 per-output-channel weight + per-token FP8-QDQ activation",
        "stage": "A: final sid_begin prefill position predicting SID-a",
        "calib_split": args.calib_split,
        "calib_sample_size": args.calib_sample_size,
        "eval_sample_size": args.eval_sample_size,
        "gptq_damp_percent": DEFAULT_GPTQ_DAMP_PERCENT,
        "gptq_block_size": DEFAULT_GPTQ_BLOCK_SIZE,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "max_new_tokens": args.max_new_tokens,
        "quantized_layer_summaries": {str(key): value.__dict__ for key, value in summaries.items()},
    }
    sample_sets: dict[str, dict[str, dict[str, Any]]] = {}
    result_paths: dict[str, str] = {}
    for mode in modes:
        print(f"[stage_a_attribution] running {mode}")
        generations: dict[str, list[str]] = {}
        start = time.time()
        with stage_a_attribution_context(model, mode):
            for sample_id, sample in tqdm(test_data.items(), desc=f"stage_a_{mode}", unit="sample"):
                generations[sample_id] = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=format_prompt(sample["prompt"], prompt_token),
                    input_device=input_device,
                    args=args,
                )
        mode_root = output_root / mode
        model_name = f"{Path(args.model_path.rstrip('/')).name}-fake-gptq-stage-a-{mode}"
        output_file = result_path(str(mode_root), model_name, args.task, args.split)
        config = {**common_config, "mode": mode}
        save_results(
            output_file=output_file,
            model_name=model_name,
            split=args.split,
            test_data=test_data,
            generations=generations,
            total_time=time.time() - start,
            config=config,
            task_name=args.task,
        )
        maybe_evaluate(str(mode_root), args.data_dir, args.overwrite, task_name=args.task)
        payload = json.loads(output_file.read_text(encoding="utf-8"))
        sample_sets[mode] = payload["samples"]
        result_paths[mode] = str(output_file)

    paired = {mode: paired_recovery(sample_sets["w8a8"], sample_sets[mode]) for mode in modes if mode != "w8a8"}
    report = {
        "result_paths": result_paths,
        "metrics": {mode: metrics_block(output_root / mode / "eval_results.json") for mode in modes},
        "paired_recovery_vs_w8a8": paired,
        "interpretation": {
            "a16": "activation-only Stage-A restoration",
            "w16": "weight-only Stage-A restoration",
            "wa16": "joint weight-and-activation Stage-A restoration",
        },
    }
    (output_root / "stage_a_attribution_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output_root / "stage_a_attribution_summary.json")


if __name__ == "__main__":
    main()
