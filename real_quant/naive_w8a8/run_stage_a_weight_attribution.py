"""Real FP8 Stage-A W8A8/A16/W16/WA16 attribution experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from fake_quant_learnable.gptq import DEFAULT_GPTQ_BLOCK_SIZE, DEFAULT_GPTQ_DAMP_PERCENT, collect_gptq_hessians
from real_quant.full_precision.results import build_generation_payload, save_generation_payload
from real_quant.full_precision.run_hf_baseline import (
    DEFAULT_DATA_DIR,
    build_output_samples,
    load_task_data,
    maybe_evaluate,
    parse_sample_size,
    resolve_repo_path,
    result_path,
)

from .apply import NaiveW8A8Summary, apply_naive_w8a8, iter_real_fp8_linears
from .gptq_runtime import advance_layer_input_batches, build_model_batches, capture_layer_input_batches, get_transformer_layers, parse_layer_indices
from .modules import RealFP8Linear, require_fp8_runtime
from .run_hf_naive_w8a8 import HFNaiveW8A8Generator
from .run_sid_stage_probe import paired_recovery
from .stage_weight_attribution_runtime import activate_stage_a_weight_attribution


DEFAULT_OUTPUT_DIR = "real_quant/naive_w8a8/results/probes/stage_a_weight_attribution"
MODES = ("w8a8", "a16", "w16", "wa16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real GPTQ Stage-A weight/activation attribution.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--task", default="ad")
    parser.add_argument("--split", default="test")
    parser.add_argument("--calib_split", default="calib")
    parser.add_argument("--calib_sample_size", default="128")
    parser.add_argument("--sample_size", default="1000")
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
    if invalid or "w8a8" not in modes:
        raise ValueError(f"Modes must include w8a8 and be among {MODES}, got {modes}")
    return modes


def _linear_snapshots(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: child.weight.detach().clone() for name, child in module.named_modules() if isinstance(child, nn.Linear)}


def _attach_snapshots(module: nn.Module, snapshots: Mapping[str, torch.Tensor]) -> None:
    for name, child in module.named_modules():
        if not isinstance(child, RealFP8Linear):
            continue
        if name not in snapshots:
            raise KeyError(f"Missing original weight for RealFP8Linear {name!r}")
        child.register_buffer("stage_probe_weight_fp", snapshots[name].detach().clone(), persistent=False)


def apply_gptq_with_weight_snapshots(
    *,
    model: nn.Module,
    model_batches: Sequence[Mapping[str, Any]],
    layer_indices: Sequence[int],
    output_dtype: torch.dtype,
) -> NaiveW8A8Summary:
    layers = get_transformer_layers(model)
    fp_inputs = None
    stream_layer_idx: int | None = None
    replaced = 0
    for layer_idx in sorted(layer_indices):
        if fp_inputs is None:
            fp_inputs = capture_layer_input_batches(model=model, layer=layers[layer_idx], model_batches=model_batches)
            stream_layer_idx = layer_idx
        else:
            assert stream_layer_idx is not None
            while stream_layer_idx < layer_idx:
                fp_inputs = advance_layer_input_batches(layer=layers[stream_layer_idx], batches=fp_inputs)
                stream_layer_idx += 1
        layer = layers[layer_idx]
        snapshots = _linear_snapshots(layer)
        hessians = collect_gptq_hessians(layer, fp_inputs)
        next_fp_inputs = advance_layer_input_batches(layer=layer, batches=fp_inputs)
        summary = apply_naive_w8a8(
            layer,
            skip_module_names=(),
            output_dtype=output_dtype,
            activation_quant_mode="dynamic",
            gptq_hessians=hessians,
            gptq_damp_percent=DEFAULT_GPTQ_DAMP_PERCENT,
            gptq_block_size=DEFAULT_GPTQ_BLOCK_SIZE,
        )
        _attach_snapshots(layer, snapshots)
        replaced += summary.replaced_linears
        fp_inputs = next_fp_inputs
        stream_layer_idx = layer_idx + 1
        print(f"[stage_a_weight_attribution] gptq layer={layer_idx} linears={summary.replaced_linears}")
    return NaiveW8A8Summary(replaced_linears=replaced, skipped_linears=0)


def metric_block(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    model = next(value for key, value in raw.items() if key != "_total_time")
    task = next(value for key, value in model.items() if key != "_total_time")
    return next(value for key, value in task.items() if key != "_total_time")


def main() -> None:
    args = parse_args()
    modes = parse_modes(args.modes)
    require_fp8_runtime()
    dtype = torch.bfloat16 if args.dtype in {"bf16", "bfloat16"} else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=dtype).to(args.device)
    model.eval()

    from benchmark.tasks.v1_0.registry import get_task_config

    task_config = get_task_config(args.task)
    prompt_token = task_config.get("generation_config", {}).get("prompt_token", "<|sid_begin|>")
    calib_data = load_task_data(task_name=args.task, data_dir=str(resolve_repo_path(args.data_dir)), tokenizer=tokenizer, split=args.calib_split, sample_size=parse_sample_size(args.calib_sample_size))
    calib_prompts = [sample["prompt"] + prompt_token if not sample["prompt"].endswith(prompt_token) else sample["prompt"] for sample in calib_data.values()]
    calib_batches = build_model_batches(tokenizer=tokenizer, prompts=calib_prompts, device=torch.device(args.device))
    summary = apply_gptq_with_weight_snapshots(model=model, model_batches=calib_batches, layer_indices=parse_layer_indices("all", num_layers=len(get_transformer_layers(model))), output_dtype=dtype)
    generator = HFNaiveW8A8Generator(model=model, tokenizer=tokenizer, model_name=f"{Path(args.model_path.rstrip('/')).name}-real-gptq-stage-a-attribution", device=args.device, num_params=float(sum(p.numel() for p in model.parameters())), quant_summary=summary)
    test_data = load_task_data(task_name=args.task, data_dir=str(resolve_repo_path(args.data_dir)), tokenizer=tokenizer, split=args.split, sample_size=parse_sample_size(args.sample_size))
    prompts = {sample_id: sample["prompt"] for sample_id, sample in test_data.items()}
    output_root = resolve_repo_path(args.output_dir)
    sample_sets: dict[str, dict[str, dict[str, Any]]] = {}
    result_paths: dict[str, str] = {}
    for mode in modes:
        print(f"[stage_a_weight_attribution] running {mode}")
        with activate_stage_a_weight_attribution(model, mode):
            generations, _ = generator.generate(prompts, prompt_token=prompt_token, batch_size=1, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams, num_return_sequences=args.num_return_sequences)
        mode_root = output_root / mode
        model_name = f"{generator.model_name}-{mode}"
        output_file = result_path(str(mode_root), model_name, args.task, args.split)
        samples = build_output_samples(test_data=test_data, generations=generations)
        payload = build_generation_payload(model_name=model_name, task_name=args.task, split=args.split, samples=samples, latency_records=list(generator.latency_records.values()), config={"probe": "real_stage_a_weight_activation_attribution", "mode": mode, "stage": "prefill final sid_begin predicts SID-a", "calib_sample_size": args.calib_sample_size, "weight_quant": "gptq"}, hardware_info=generator.get_hardware_info(), num_params=generator.num_params)
        save_generation_payload(payload, output_file)
        maybe_evaluate(str(mode_root), args.data_dir, args.overwrite, task_name=args.task)
        sample_sets[mode] = samples
        result_paths[mode] = str(output_file)
    result = {"result_paths": result_paths, "metrics": {mode: metric_block(output_root / mode / "eval_results.json") for mode in modes}, "paired_recovery_vs_w8a8": {mode: paired_recovery(sample_sets["w8a8"], sample_sets[mode]) for mode in modes if mode != "w8a8"}}
    (output_root / "stage_a_real_attribution_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output_root / "stage_a_real_attribution_summary.json")


if __name__ == "__main__":
    main()
