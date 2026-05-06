#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark import Benchmark
from benchmark.tasks.v1_0.registry import get_loader, get_task_config

try:
    from .apply import apply_fp8_fake_quant
except ImportError:
    from fake_quant.apply import apply_fp8_fake_quant


DEFAULT_MODEL_PATH = "/home/guowei/OneRec-1.7B"
DEFAULT_DATA_DIR = "../data/onerec_data/benchmark_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HF AD SID prediction with optional FP8 fake quantization.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="HF base model path.")
    parser.add_argument("--model_name", default=None, help="Name used in result directory and JSON.")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default="results/v1.0/results_OneRec-1.7B-hf-fake-fp8")
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--sample_size", default=None, help='Integer sample size, "full", or omitted for task default.')
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device_map", default=None, help='Optional HF device_map, e.g. "auto".')
    parser.add_argument("--quant_scheme", default="fp8_weight_channel", choices=["none", "fp8_weight_channel"])
    parser.add_argument("--act_quant", default="none", choices=["none", "per_token"])
    parser.add_argument(
        "--target_regex",
        default=None,
        help="Optional regex over module names. Default quantizes all Linear modules except skipped names.",
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


def generate_one(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    input_device: torch.device,
    args: argparse.Namespace,
) -> List[str]:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
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
    return Path(output_dir) / model_name / "ad" / f"{split}_generated.json"


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
            "target_regex": args.target_regex,
            "quantize_lm_head": args.quantize_lm_head,
            "dtype": args.dtype,
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_return_sequences,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "samples": samples,
    }
    output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def maybe_evaluate(output_dir: str, data_dir: str, overwrite: bool) -> None:
    Benchmark.evaluate_dev(
        generation_results_dir=output_dir,
        output_path=str(Path(output_dir) / "eval_results.json"),
        data_dir=data_dir,
        overwrite=overwrite,
        task_types=["ad"],
    )


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("The first-pass HF fake-quant runner currently supports --batch_size 1 only.")

    set_seed(args.seed)

    model_name = args.model_name or Path(args.model_path.rstrip("/")).name
    output_file = result_path(args.output_dir, model_name, args.split)
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
            skip_module_names=skip_names,
            target_regex=args.target_regex,
        )
        print(
            f"Applied FP8 fake quant: replaced_linears={summary.replaced_linears}, "
            f"skipped_linears={summary.skipped_linears}, act_quant={args.act_quant}"
        )
    elif args.act_quant != "none":
        raise ValueError("--act_quant requires --quant_scheme fp8_weight_channel in this runner.")

    input_device = resolve_input_device(model, args.device)
    task_config = get_task_config("ad")
    prompt_token = task_config.get("generation_config", {}).get("prompt_token", "<|sid_begin|>")
    sample_size = parse_sample_size(args.sample_size)
    test_data = load_ad_data(tokenizer, args.data_dir, args.split, sample_size)

    generations: Dict[str, List[str]] = {}
    start = time.time()
    for sample_id, sample in tqdm(test_data.items(), desc="HF fake-quant AD generation"):
        prompt = sample["prompt"]
        if prompt_token and not prompt.endswith(prompt_token):
            prompt = prompt + prompt_token
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
        args=args,
    )
    print(f"Generation results saved to: {output_file}")
    print(f"Total time: {total_time:.2f}s, avg/sample: {total_time / len(test_data):.4f}s")

    if args.evaluate:
        maybe_evaluate(args.output_dir, args.data_dir, args.overwrite)


if __name__ == "__main__":
    main()
