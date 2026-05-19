#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
for path in (PROJECT_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark import Benchmark
from benchmark.tasks.v1_0.registry import get_loader, get_task_config

try:
    from .apply import apply_fp8_fake_quant, apply_smoothquant_fp8_fake_quant
except ImportError:
    from fake_quant.apply import apply_fp8_fake_quant, apply_smoothquant_fp8_fake_quant


DEFAULT_MODEL_PATH = "/home/guowei/OneRec-1.7B"
DEFAULT_DATA_DIR = "data/onerec_data/benchmark-data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HF AD SID prediction with optional FP8 fake quantization.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="HF base model path.")
    parser.add_argument("--model_name", default=None, help="Name used in result directory and JSON.")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--eval_data_dir",
        default=None,
        help="Optional data_dir passed to Benchmark.evaluate_dev. Defaults to --data_dir.",
    )
    parser.add_argument("--output_dir", default="results/v1.0/results_OneRec-1.7B-hf-fake-fp8")
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--sample_size", default=None, help='Integer sample size, "full", or omitted for task default.')
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default=None, help='Optional HF device_map, e.g. "auto".')
    parser.add_argument(
        "--quant_scheme",
        default="fp8_weight_channel",
        choices=["none", "fp8_weight_channel", "fp8_smoothquant"],
    )
    parser.add_argument("--act_quant", default="none", choices=["none", "per_token", "static_tensor"])
    parser.add_argument(
        "--act_quant_mode",
        default="per_linear",
        choices=["per_linear", "shared_input"],
        help="per_linear quantizes each Linear input independently; shared_input reuses qkv/gate-up activation QDQ.",
    )
    parser.add_argument(
        "--target_regex",
        default=None,
        help="Optional regex over module names. Default quantizes all Linear modules except skipped names.",
    )
    parser.add_argument(
        "--skip_regex",
        default=None,
        help="Optional regex over module names to skip from fake quantization.",
    )
    parser.add_argument(
        "--smooth_scales_path",
        default=None,
        help="Path to SmoothQuant calibration absmax file produced by collect_smooth_scales.py.",
    )
    parser.add_argument(
        "--static_act_scales_path",
        default=None,
        help="Path to calibrated activation absmax file for act_quant=static_tensor.",
    )
    parser.add_argument("--smooth_alpha", type=float, default=0.5, help="SmoothQuant alpha in [0, 1].")
    parser.add_argument(
        "--smooth_layer_cutoff",
        type=int,
        default=None,
        help=(
            "Apply SmoothQuant smoothing only to layers with layer_idx < cutoff. "
            "Higher layers still use plain FP8 fake quant without smoothing."
        ),
    )
    parser.add_argument(
        "--smooth_layer_min",
        type=int,
        default=None,
        help=(
            "Apply SmoothQuant smoothing only to layers with layer_idx >= min. "
            "Lower layers still use plain FP8 fake quant without smoothing."
        ),
    )
    parser.add_argument("--quantize_lm_head", action="store_true")
    parser.add_argument("--num_beams", type=int, default=32)
    parser.add_argument("--num_return_sequences", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size. Keep 1 for first-pass exact slicing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Run benchmark evaluator after generation.")
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_input_device(model: torch.nn.Module, fallback: str) -> torch.device:
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


def load_ad_data(tokenizer: Any, data_dir: str, split: str, sample_size: Any) -> Dict[str, Dict[str, Any]]:
    loader = get_loader(
        task_name="ad",
        data_dir=data_dir,
        enable_thinking=False,
        tokenizer=tokenizer,
    )
    return loader.load_data(split=split, sample_size=sample_size)


def decode_generations(
    tokenizer: Any,
    sequences: torch.Tensor,
    prompt_len: int,
) -> List[str]:
    generations = []
    for seq in sequences:
        generated_ids = seq[prompt_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        generations.append(text)
    return generations


def iter_batches(
    items: List[Tuple[str, Dict[str, Any]]],
    batch_size: int,
) -> Iterator[List[Tuple[str, Dict[str, Any]]]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def format_prompt(prompt: str, prompt_token: str) -> str:
    if prompt_token and not prompt.endswith(prompt_token):
        return prompt + prompt_token
    return prompt


def generate_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: List[str],
    input_device: torch.device,
    args: argparse.Namespace,
) -> List[List[str]]:
    if not prompts:
        return []

    old_padding_side = getattr(tokenizer, "padding_side", None)
    if old_padding_side is not None:
        tokenizer.padding_side = "left"

    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
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
    finally:
        if old_padding_side is not None:
            tokenizer.padding_side = old_padding_side

    output = output.detach().cpu()
    if output.shape[0] % len(prompts) != 0:
        raise RuntimeError(
            "Unexpected generate output shape: "
            f"{tuple(output.shape)} for batch size {len(prompts)}"
        )
    returns_per_prompt = output.shape[0] // len(prompts)
    generations = []
    for batch_idx in range(len(prompts)):
        start = batch_idx * returns_per_prompt
        end = start + returns_per_prompt
        generations.append(decode_generations(tokenizer, output[start:end], prompt_len))
    return generations


def generate_one(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    input_device: torch.device,
    args: argparse.Namespace,
) -> List[str]:
    return generate_batch(model, tokenizer, [prompt], input_device, args)[0]


def result_path(output_dir: str, model_name: str, split: str) -> Path:
    return resolve_repo_path(output_dir) / model_name / "ad" / f"{split}_generated.json"


def save_results(
    *,
    output_file: Path,
    model_name: str,
    split: str,
    test_data: Dict[str, Dict[str, Any]],
    generations: Dict[str, List[str]],
    total_time: float,
    args: argparse.Namespace,
) -> None:
    samples = {}
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
    data = {
        "model_name": model_name,
        "task_name": "ad",
        "split": split,
        "total_time": total_time,
        "avg_time_per_sample": total_time / len(samples) if samples else 0.0,
        "fake_quant_config": {
            "quant_scheme": args.quant_scheme,
            "act_quant": args.act_quant,
            "act_quant_mode": args.act_quant_mode,
            "target_regex": args.target_regex,
            "skip_regex": args.skip_regex,
            "smooth_scales_path": args.smooth_scales_path,
            "static_act_scales_path": args.static_act_scales_path,
            "smooth_alpha": args.smooth_alpha,
            "smooth_layer_cutoff": args.smooth_layer_cutoff,
            "smooth_layer_min": args.smooth_layer_min,
            "quantize_lm_head": args.quantize_lm_head,
            "dtype": args.dtype,
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_return_sequences,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "samples": samples,
    }
    output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be positive, got {args.batch_size}")

    set_seed(args.seed)

    model_name = args.model_name or Path(args.model_path.rstrip("/")).name
    output_file = result_path(args.output_dir, model_name, args.split)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Generation file exists: {output_file}. Use --overwrite to regenerate.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": dtype_from_name(args.dtype),
        "trust_remote_code": True,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    if not args.device_map:
        model = model.to(args.device)
    model.eval()

    if args.quant_scheme == "fp8_weight_channel":
        skip_names: Tuple[str, ...] = () if args.quantize_lm_head else ("lm_head",)
        summary = apply_fp8_fake_quant(
            model,
            act_quant=args.act_quant,
            act_quant_mode=args.act_quant_mode,
            activation_absmax_path=args.static_act_scales_path,
            skip_module_names=skip_names,
            target_regex=args.target_regex,
            skip_regex=args.skip_regex,
        )
        print(
            f"Applied FP8 fake quant: replaced_linears={summary.replaced_linears}, "
            f"skipped_linears={summary.skipped_linears}, act_quant={args.act_quant}, "
            f"act_quant_mode={args.act_quant_mode}, "
            f"static_act_scales_path={args.static_act_scales_path}, "
            f"shared_attention_modules={summary.shared_attention_modules}, "
            f"shared_mlp_modules={summary.shared_mlp_modules}"
        )
    elif args.quant_scheme == "fp8_smoothquant":
        if not args.smooth_scales_path:
            raise ValueError("--quant_scheme fp8_smoothquant requires --smooth_scales_path.")
        skip_names = () if args.quantize_lm_head else ("lm_head",)
        summary = apply_smoothquant_fp8_fake_quant(
            model,
            activation_absmax_path=args.smooth_scales_path,
            alpha=args.smooth_alpha,
            smooth_layer_min=args.smooth_layer_min,
            smooth_layer_cutoff=args.smooth_layer_cutoff,
            act_quant=args.act_quant,
            act_quant_mode=args.act_quant_mode,
            skip_module_names=skip_names,
            target_regex=args.target_regex,
            skip_regex=args.skip_regex,
        )
        print(
            f"Applied SmoothQuant FP8 fake quant: replaced_linears={summary.replaced_linears}, "
            f"skipped_linears={summary.skipped_linears}, act_quant={args.act_quant}, "
            f"act_quant_mode={args.act_quant_mode}, smooth_alpha={args.smooth_alpha}, "
            f"smooth_scales_path={args.smooth_scales_path}, "
            f"smooth_layer_min={args.smooth_layer_min}, "
            f"smooth_layer_cutoff={args.smooth_layer_cutoff}, "
            f"shared_attention_modules={summary.shared_attention_modules}, "
            f"shared_mlp_modules={summary.shared_mlp_modules}"
        )
    elif args.act_quant != "none":
        raise ValueError("--act_quant requires a fake-quant --quant_scheme in this runner.")

    input_device = resolve_input_device(model, args.device)
    task_config = get_task_config("ad")
    prompt_token = task_config.get("generation_config", {}).get("prompt_token", "<|sid_begin|>")
    sample_size = parse_sample_size(args.sample_size)
    test_data = load_ad_data(tokenizer, str(resolve_repo_path(args.data_dir)), args.split, sample_size)

    generations: Dict[str, List[str]] = {}
    start = time.time()
    test_items = list(test_data.items())
    num_batches = (len(test_items) + args.batch_size - 1) // args.batch_size
    for batch in tqdm(
        iter_batches(test_items, args.batch_size),
        desc="HF fake-quant AD generation",
        total=num_batches,
    ):
        sample_ids = [sample_id for sample_id, _sample in batch]
        prompts = [format_prompt(sample["prompt"], prompt_token) for _sample_id, sample in batch]
        batch_generations = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            input_device=input_device,
            args=args,
        )
        for sample_id, sample_generations in zip(sample_ids, batch_generations):
            generations[sample_id] = sample_generations
    total_time = time.time() - start

    save_results(
        output_file=output_file,
        model_name=model_name,
        split=args.split,
        test_data=test_data,
        generations=generations,
        total_time=total_time,
        args=args,
    )
    print(f"Generation results saved to: {output_file}")
    print(f"Total time: {total_time:.2f}s, avg/sample: {total_time / len(test_data):.4f}s")

    if args.evaluate:
        maybe_evaluate(args.output_dir, args.eval_data_dir or args.data_dir, args.overwrite)


if __name__ == "__main__":
    main()
