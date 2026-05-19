#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
for path in (PROJECT_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.tasks.v1_0.registry import get_loader, get_task_config

try:
    from .core import save_activation_absmax
except ImportError:
    from fake_quant.smoothquant.core import save_activation_absmax


DEFAULT_MODEL_PATH = "/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B"
DEFAULT_DATA_DIR = "data/onerec_data/benchmark-data"
DEFAULT_OUTPUT_PATH = "fake_quant/smoothquant/scales/onerec_ad_smoothquant_absmax_sample128.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect SmoothQuant activation absmax stats on AD prompts.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--sample_size", default=128, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument(
        "--sample_offset",
        default=0,
        type=int,
        help="Start offset within the split. Use 1000 to avoid overlap with eval sample1000.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--target_regex", default=None)
    parser.add_argument("--skip_regex", default=None)
    parser.add_argument("--collect_lm_head", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


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


def slice_calibration_data(
    data: Dict[str, Dict[str, Any]],
    *,
    sample_size: int,
    sample_offset: int,
) -> Dict[str, Dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}")
    if sample_offset < 0:
        raise ValueError(f"sample_offset must be non-negative, got {sample_offset}")

    items = list(data.items())
    end = sample_offset + sample_size
    if end > len(items):
        raise ValueError(
            f"Not enough samples for calibration slice: offset={sample_offset}, "
            f"sample_size={sample_size}, total={len(items)}"
        )
    return dict(items[sample_offset:end])


def load_ad_data(
    tokenizer: Any,
    data_dir: str,
    split: str,
    sample_size: int,
    sample_offset: int = 0,
) -> Dict[str, Dict[str, Any]]:
    loader = get_loader(
        task_name="ad",
        data_dir=data_dir,
        enable_thinking=False,
        tokenizer=tokenizer,
    )
    if sample_offset > 0:
        data = loader.load_data(split=split, sample_size="full")
        return slice_calibration_data(
            data,
            sample_size=sample_size,
            sample_offset=sample_offset,
        )
    return loader.load_data(split=split, sample_size=sample_size)


def should_collect_module(
    name: str,
    *,
    collect_lm_head: bool,
    target_pattern: re.Pattern[str] | None,
    skip_pattern: re.Pattern[str] | None,
) -> bool:
    child_name = name.rsplit(".", 1)[-1]
    if not collect_lm_head and (child_name == "lm_head" or name == "lm_head"):
        return False
    if skip_pattern is not None and skip_pattern.search(name) is not None:
        return False
    if target_pattern is not None and target_pattern.search(name) is None:
        return False
    return True


def iter_sample_batches(
    samples: list[Dict[str, Any]],
    batch_size: int,
) -> Iterator[list[Dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for start in range(0, len(samples), batch_size):
        yield samples[start : start + batch_size]


def format_prompt(prompt: str, prompt_token: str) -> str:
    if prompt_token and not prompt.endswith(prompt_token):
        return prompt + prompt_token
    return prompt


def collect_activation_absmax(
    model: nn.Module,
    tokenizer: Any,
    test_data: Dict[str, Dict[str, Any]],
    *,
    input_device: torch.device,
    prompt_token: str,
    collect_lm_head: bool,
    target_regex: str | None,
    skip_regex: str | None,
    batch_size: int = 1,
) -> Dict[str, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    stats: Dict[str, torch.Tensor] = {}
    handles = []
    target_pattern = re.compile(target_regex) if target_regex else None
    skip_pattern = re.compile(skip_regex) if skip_regex else None
    current_attention_mask: torch.Tensor | None = None

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            nonlocal current_attention_mask
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            x_for_stats = x.detach()
            if (
                current_attention_mask is not None
                and x_for_stats.ndim >= 3
                and tuple(x_for_stats.shape[:2]) == tuple(current_attention_mask.shape)
            ):
                mask = current_attention_mask.to(device=x_for_stats.device, dtype=torch.bool)
                view_shape = (*mask.shape, *((1,) * (x_for_stats.ndim - 2)))
                x_for_stats = x_for_stats.masked_fill(~mask.view(view_shape), 0)
            reduce_dims = tuple(range(x_for_stats.ndim - 1))
            x_absmax = x_for_stats.float().abs().amax(dim=reduce_dims).cpu()
            if name in stats:
                stats[name] = torch.maximum(stats[name], x_absmax)
            else:
                stats[name] = x_absmax

        return hook

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if should_collect_module(
            name,
            collect_lm_head=collect_lm_head,
            target_pattern=target_pattern,
            skip_pattern=skip_pattern,
        ):
            handles.append(module.register_forward_pre_hook(make_hook(name)))

    old_padding_side = getattr(tokenizer, "padding_side", None)
    if old_padding_side is not None:
        tokenizer.padding_side = "left"

    try:
        with torch.inference_mode():
            samples = list(test_data.values())
            num_batches = (len(samples) + batch_size - 1) // batch_size
            for batch in tqdm(
                iter_sample_batches(samples, batch_size),
                desc="Collect SmoothQuant activation absmax",
                total=num_batches,
            ):
                prompts = [format_prompt(sample["prompt"], prompt_token) for sample in batch]
                inputs = tokenizer(prompts, return_tensors="pt", padding=True)
                inputs = {
                    key: value.to(input_device) if torch.is_tensor(value) else value
                    for key, value in inputs.items()
                }
                current_attention_mask = inputs.get("attention_mask")
                model(**inputs, use_cache=False)
                current_attention_mask = None
    finally:
        if old_padding_side is not None:
            tokenizer.padding_side = old_padding_side
        for handle in handles:
            handle.remove()

    if not stats:
        raise RuntimeError("No activation stats were collected. Check target_regex/skip_regex.")
    return stats


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

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

    input_device = resolve_input_device(model, args.device)
    task_config = get_task_config("ad")
    prompt_token = task_config.get("generation_config", {}).get("prompt_token", "<|sid_begin|>")
    test_data = load_ad_data(
        tokenizer,
        args.data_dir,
        args.split,
        args.sample_size,
        sample_offset=args.sample_offset,
    )
    stats = collect_activation_absmax(
        model,
        tokenizer,
        test_data,
        input_device=input_device,
        prompt_token=prompt_token,
        collect_lm_head=args.collect_lm_head,
        target_regex=args.target_regex,
        skip_regex=args.skip_regex,
        batch_size=args.batch_size,
    )

    save_activation_absmax(
        args.output_path,
        activation_absmax=stats,
        metadata={
            "model_path": args.model_path,
            "data_dir": args.data_dir,
            "split": args.split,
            "sample_size": args.sample_size,
            "batch_size": args.batch_size,
            "sample_offset": args.sample_offset,
            "sample_range": [args.sample_offset, args.sample_offset + args.sample_size],
            "dtype": args.dtype,
            "seed": args.seed,
            "target_regex": args.target_regex,
            "skip_regex": args.skip_regex,
            "collect_lm_head": args.collect_lm_head,
        },
    )
    print(f"Saved SmoothQuant activation absmax stats to: {args.output_path}")
    print(f"Collected Linear modules: {len(stats)}")


if __name__ == "__main__":
    main()
