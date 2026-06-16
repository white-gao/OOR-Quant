#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.tasks.v1_0.registry import get_task_config
from fake_quant_learnable.apply import BaselineQuantSummary
from fake_quant_learnable.run_m1_onerec_ad import (
    DEFAULT_ACT_QUANT,
    DEFAULT_ACT_QUANT_MODE,
    DEFAULT_DATA_DIR,
    DEFAULT_DTYPE,
    DEFAULT_EVAL_OFFSET,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_PATH,
    DEFAULT_NUM_BEAMS,
    DEFAULT_NUM_RETURN_SEQUENCES,
    DEFAULT_SEED,
    DEFAULT_SPLIT,
    dtype_from_name,
    format_prompt,
    generate_one,
    get_transformer_layers,
    load_ad_data,
    maybe_evaluate,
    parse_layer_indices,
    parse_sample_size,
    resolve_input_device,
    resolve_repo_path,
    result_path,
    save_results,
    set_seed,
    summaries_to_jsonable,
)
from fake_quant_learnable.support.ablation.decode_a16 import (
    DecodeA16BaselineFakeQuantLinear,
    apply_decode_a16_w8a8,
)


DEFAULT_OUTPUT_DIR = "fake_quant_learnable/results/w8a8_decode_a16_ad"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AD W8A8 baseline with decode-step activation A16 ablation."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_sample_size", default="full")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num_beams", type=int, default=DEFAULT_NUM_BEAMS)
    parser.add_argument("--num_return_sequences", type=int, default=DEFAULT_NUM_RETURN_SEQUENCES)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def apply_decode_a16_layers(
    *,
    model: nn.Module,
    layer_indices: list[int],
) -> dict[int, BaselineQuantSummary]:
    layers = get_transformer_layers(model)
    summaries: dict[int, BaselineQuantSummary] = {}
    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        if isinstance(layer, nn.Linear):
            layers[layer_idx] = DecodeA16BaselineFakeQuantLinear(layer, act_quant=DEFAULT_ACT_QUANT)
            summary = BaselineQuantSummary(replaced_linears=1, skipped_linears=0)
        else:
            summary = apply_decode_a16_w8a8(
                layer,
                act_quant=DEFAULT_ACT_QUANT,
                act_quant_mode=DEFAULT_ACT_QUANT_MODE,
            )
        summaries[layer_idx] = summary
        print(
            f"[w8a8_decode_a16] layer={layer_idx} replaced_linears={summary.replaced_linears} "
            f"shared_attention_modules={summary.shared_attention_modules} "
            f"shared_mlp_modules={summary.shared_mlp_modules}"
        )
    return summaries


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model_name = Path(args.model_path.rstrip("/")).name
    output_file = result_path(args.output_dir, model_name, DEFAULT_SPLIT)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Generation file exists: {output_file}. Use --overwrite.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_from_name(args.dtype),
        trust_remote_code=True,
    ).to(args.device)
    model.eval()
    input_device = resolve_input_device(model, args.device)

    task_config = get_task_config("ad")
    prompt_token = task_config.get("generation_config", {}).get("prompt_token", "<|sid_begin|>")

    layers = get_transformer_layers(model)
    layer_indices = parse_layer_indices(args.layers, num_layers=len(layers))
    baseline_summaries = apply_decode_a16_layers(
        model=model,
        layer_indices=layer_indices,
    )

    config: dict[str, Any] = {
        "method": "w8a8_decode_a16",
        "definition": "prefill activation W8A8 for all prompt tokens; single-token decode activation bypasses A8; weights stay W8",
        "layers": layer_indices,
        "act_quant": DEFAULT_ACT_QUANT,
        "act_quant_mode": DEFAULT_ACT_QUANT_MODE,
        "data_dir": args.data_dir,
        "split": DEFAULT_SPLIT,
        "eval_sample_size": args.eval_sample_size,
        "eval_offset": DEFAULT_EVAL_OFFSET,
        "dtype": args.dtype,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "baseline_summaries": summaries_to_jsonable(baseline_summaries),
    }
    (output_file.parent / "w8a8_decode_a16_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    test_data = load_ad_data(
        tokenizer,
        str(resolve_repo_path(args.data_dir)),
        DEFAULT_SPLIT,
        parse_sample_size(args.eval_sample_size),
        sample_offset=DEFAULT_EVAL_OFFSET,
    )
    test_items = list(test_data.items())
    generations: dict[str, list[str]] = {}
    start = time.time()
    for sample_id, sample in tqdm(
        test_items,
        total=len(test_items),
        desc="w8a8_decode_a16 AD generation",
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
        split=DEFAULT_SPLIT,
        test_data=test_data,
        generations=generations,
        total_time=total_time,
        config=config,
    )
    if args.evaluate:
        maybe_evaluate(args.output_dir, args.data_dir, args.overwrite)


if __name__ == "__main__":
    main()
