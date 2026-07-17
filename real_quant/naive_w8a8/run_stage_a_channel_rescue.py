"""Causal test for concentrated Stage-A activation-QDQ channel error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

from .gptq_runtime import build_model_batches, get_transformer_layers, parse_layer_indices
from .modules import require_fp8_runtime
from .run_hf_naive_w8a8 import HFNaiveW8A8Generator
from .run_sid_stage_probe import paired_recovery
from .run_stage_a_weight_attribution import apply_gptq_with_weight_snapshots, metric_block
from .stage_channel_rescue import activate_stage_a_channel_rescue


DEFAULT_OUTPUT_DIR = "real_quant/naive_w8a8/results/probes/stage_a_channel_rescue"
DEFAULT_ERROR_VECTORS = "real_quant/naive_w8a8/results/probes/stage_a_activation_error_ad_1p7b_gptq_calib128/stage_a_activation_error_vectors.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real Stage-A selective-channel activation rescue.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--task", default="ad")
    parser.add_argument("--split", default="test")
    parser.add_argument("--calib_split", default="calib")
    parser.add_argument("--calib_sample_size", default="128")
    parser.add_argument("--sample_size", default="1000")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--error_vectors", default=DEFAULT_ERROR_VECTORS)
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--random_seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_beams", type=int, default=32)
    parser.add_argument("--num_return_sequences", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def channel_sets(vectors: dict[str, torch.Tensor], fraction: float, seed: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    error: dict[str, torch.Tensor] = {}
    random: dict[str, torch.Tensor] = {}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for key, vector in vectors.items():
        if not key.endswith(".mean_error2"):
            continue
        label = key.removesuffix(".mean_error2")
        count = max(1, int(round(vector.numel() * fraction)))
        error[label] = torch.topk(vector.float(), k=count).indices.cpu()
        random[label] = torch.randperm(vector.numel(), generator=generator)[:count]
    return error, random


def main() -> None:
    args = parse_args()
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    require_fp8_runtime()
    vectors = torch.load(resolve_repo_path(args.error_vectors), map_location="cpu", weights_only=False)
    error_indices, random_indices = channel_sets(vectors, args.fraction, args.random_seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(args.device)
    model.eval()
    from benchmark.tasks.v1_0.registry import get_task_config
    prompt_token = get_task_config(args.task).get("generation_config", {}).get("prompt_token", "<|sid_begin|>")
    calib_data = load_task_data(task_name=args.task, data_dir=str(resolve_repo_path(args.data_dir)), tokenizer=tokenizer, split=args.calib_split, sample_size=parse_sample_size(args.calib_sample_size))
    calib_prompts = [sample["prompt"] + prompt_token if not sample["prompt"].endswith(prompt_token) else sample["prompt"] for sample in calib_data.values()]
    calib_batches = build_model_batches(tokenizer=tokenizer, prompts=calib_prompts, device=torch.device(args.device))
    summary = apply_gptq_with_weight_snapshots(model=model, model_batches=calib_batches, layer_indices=parse_layer_indices("all", num_layers=len(get_transformer_layers(model))), output_dtype=torch.bfloat16)
    generator = HFNaiveW8A8Generator(model=model, tokenizer=tokenizer, model_name=f"{Path(args.model_path.rstrip('/')).name}-real-gptq-stage-a-channel", device=args.device, num_params=float(sum(p.numel() for p in model.parameters())), quant_summary=summary)
    test_data = load_task_data(task_name=args.task, data_dir=str(resolve_repo_path(args.data_dir)), tokenizer=tokenizer, split=args.split, sample_size=parse_sample_size(args.sample_size))
    prompts = {sample_id: sample["prompt"] for sample_id, sample in test_data.items()}
    variants = {"w8a8": None, "error_top1pct": error_indices, "random_top1pct": random_indices}
    root = resolve_repo_path(args.output_dir)
    sample_sets, paths = {}, {}
    for name, indices in variants.items():
        print(f"[stage_a_channel_rescue] running {name}")
        context = activate_stage_a_channel_rescue(model, indices) if indices is not None else torch.no_grad()
        with context:
            generations, _ = generator.generate(prompts, prompt_token=prompt_token, batch_size=1, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams, num_return_sequences=args.num_return_sequences)
        samples = build_output_samples(test_data=test_data, generations=generations)
        out_file = result_path(str(root / name), f"{generator.model_name}-{name}", args.task, args.split)
        payload = build_generation_payload(model_name=f"{generator.model_name}-{name}", task_name=args.task, split=args.split, samples=samples, latency_records=list(generator.latency_records.values()), config={"probe": "stage_a_selective_channel_rescue", "variant": name, "fraction": args.fraction}, hardware_info=generator.get_hardware_info(), num_params=generator.num_params)
        save_generation_payload(payload, out_file)
        maybe_evaluate(str(root / name), args.data_dir, args.overwrite, task_name=args.task)
        sample_sets[name], paths[name] = samples, str(out_file)
    output = {"fraction": args.fraction, "result_paths": paths, "metrics": {name: metric_block(root / name / "eval_results.json") for name in variants}, "paired_recovery_vs_w8a8": {name: paired_recovery(sample_sets["w8a8"], sample_sets[name]) for name in variants if name != "w8a8"}}
    (root / "stage_a_channel_rescue_summary.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(root / "stage_a_channel_rescue_summary.json")


if __name__ == "__main__":
    main()
