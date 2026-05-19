#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
for path in (PROJECT_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.tasks.v1_0.registry import get_task_config

from fake_quant.probes.activation_probe.profile_ad_activations import (
    ActivationProfiler,
    ForwardContext,
    aggregate_rows,
    encode_prompt,
    load_ad_data,
    parse_sample_size,
    parse_thresholds,
    register_hooks,
    resolve_data_dir,
    resolve_model_path,
    torch_dtype,
    write_csv,
    write_json,
)


DEFAULT_MODEL_PATH = "/home/guowei/OneRec-1.7B"
DEFAULT_DATA_DIR = "data/onerec_data/benchmark-data"
DEFAULT_OUTPUT_DIR = (
    "fake_quant/probes/activation_probe/activation_profiles/v1.0/"
    "OneRec-1.7B-ad-teacher-forced-sample-128"
)
SID_PATTERN = re.compile(
    r"<\|sid_begin\|>(<s_a_\d+>)(<s_b_\d+>)(<s_c_\d+>)<\|sid_end\|>"
)
STAGES = ["predict_a", "predict_b", "predict_c", "predict_end"]
INPUT_GROUPS = ["input_sid_begin", "input_a_sid_code", "input_b_sid_code", "input_c_sid_code"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile fixed-position OneRec AD activations with teacher-forced SID decoding."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample_size", default="128")
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max_tokens", type=int, default=0, help="Optional left truncation for long prompts.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--outlier_thresholds",
        default="6,10,20",
        help="Comma-separated absolute activation thresholds for outlier ratios.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def first_ground_truth_sid(ground_truth: str) -> List[str]:
    match = SID_PATTERN.search(ground_truth or "")
    if match is None:
        raise ValueError(f"No SID sequence found in ground_truth: {ground_truth!r}")
    return [match.group(1), match.group(2), match.group(3), "<|sid_end|>"]


def token_id_for_text(tokenizer: Any, token_text: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token_text)
    if token_id is not None and token_id != tokenizer.unk_token_id:
        return int(token_id)

    encoded = tokenizer(token_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    if len(encoded) != 1:
        raise ValueError(f"Expected {token_text!r} to map to one token, got ids={encoded}")
    return int(encoded[0])


def target_rank(logits: torch.Tensor, target_id: int) -> int:
    target_logit = logits[0, target_id]
    return int((logits[0] > target_logit).sum().item() + 1)


def profile_teacher_forced_sample(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    profiler: ActivationProfiler,
    sample_id: str,
    prompt: str,
    ground_truth: str,
    prompt_token: str,
    device: str,
    max_tokens: int,
) -> Dict[str, Any]:
    target_texts = first_ground_truth_sid(ground_truth)
    target_ids = [token_id_for_text(tokenizer, token_text) for token_text in target_texts]

    input_ids = encode_prompt(tokenizer, prompt, prompt_token, max_tokens).to(device)
    prompt_tokens = int(input_ids.shape[1])

    step_summaries: List[Dict[str, Any]] = []
    past_key_values = None

    with torch.inference_mode():
        profiler.context = ForwardContext(
            sample_id=sample_id,
            stage="teacher_prefill",
            token_groups=["prompt_token"] * max(prompt_tokens - 1, 0) + [INPUT_GROUPS[0]],
            collect_all_groups=False,
            last_token_stage=STAGES[0],
            last_token_group=INPUT_GROUPS[0],
        )
        output = model(input_ids=input_ids, use_cache=True)
        logits = output.logits[:, -1, :]
        step_summaries.append(
            {
                "stage": STAGES[0],
                "input_group": INPUT_GROUPS[0],
                "target_text": target_texts[0],
                "target_id": target_ids[0],
                "target_rank": target_rank(logits.detach().float().cpu(), target_ids[0]),
            }
        )
        past_key_values = output.past_key_values

        for index in range(1, len(STAGES)):
            forced_input_id = target_ids[index - 1]
            target_id = target_ids[index]
            input_token = torch.tensor([[forced_input_id]], device=device)
            profiler.context = ForwardContext(
                sample_id=sample_id,
                stage=STAGES[index],
                token_groups=[INPUT_GROUPS[index]],
                collect_all_groups=False,
            )
            output = model(input_ids=input_token, past_key_values=past_key_values, use_cache=True)
            logits = output.logits[:, -1, :]
            step_summaries.append(
                {
                    "stage": STAGES[index],
                    "input_group": INPUT_GROUPS[index],
                    "forced_input_text": target_texts[index - 1],
                    "forced_input_id": forced_input_id,
                    "target_text": target_texts[index],
                    "target_id": target_id,
                    "target_rank": target_rank(logits.detach().float().cpu(), target_id),
                }
            )
            past_key_values = output.past_key_values

    profiler.context = None
    return {
        "sample_id": sample_id,
        "prompt_tokens": prompt_tokens,
        "target_sid": target_texts,
        "steps": step_summaries,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    output_dir = Path(args.output_dir)
    thresholds = parse_thresholds(args.outlier_thresholds)

    model_path = resolve_model_path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype(args.dtype),
        device_map=None,
    ).to(args.device)
    model.eval()

    sample_size = parse_sample_size(args.sample_size)
    data_dir = resolve_data_dir(args.data_dir)
    test_data = load_ad_data(tokenizer, data_dir, args.split, sample_size)
    prompt_token = get_task_config("ad").get("generation_config", {}).get("prompt_token", "<|sid_begin|>")

    profiler = ActivationProfiler(thresholds)
    register_hooks(model, profiler)

    start = time.time()
    sample_summaries = []
    skipped_samples: List[Dict[str, str]] = []
    try:
        for sample_id, sample in tqdm(test_data.items(), desc="Teacher-forced AD activations"):
            try:
                sample_summaries.append(
                    profile_teacher_forced_sample(
                        model=model,
                        tokenizer=tokenizer,
                        profiler=profiler,
                        sample_id=sample_id,
                        prompt=sample["prompt"],
                        ground_truth=sample.get("ground_truth", ""),
                        prompt_token=prompt_token,
                        device=args.device,
                        max_tokens=args.max_tokens,
                    )
                )
            except ValueError as exc:
                skipped_samples.append({"sample_id": sample_id, "reason": str(exc)})
    finally:
        profiler.close()

    elapsed = time.time() - start
    rows = profiler.rows
    write_csv(output_dir / "event_stats.csv", rows)
    write_csv(output_dir / "summary_by_step_layer_module.csv", aggregate_rows(rows, ["stage", "layer", "module"]))
    write_csv(output_dir / "summary_by_step_module.csv", aggregate_rows(rows, ["stage", "module"]))
    write_csv(
        output_dir / "summary_by_step_input_group_module.csv",
        aggregate_rows(rows, ["stage", "token_group", "module"]),
    )
    write_json(
        output_dir / "sample_summary.json",
        {
            "config": vars(args),
            "resolved_model_path": model_path,
            "resolved_data_dir": data_dir,
            "num_samples": len(test_data),
            "num_profiled_samples": len(sample_summaries),
            "num_skipped_samples": len(skipped_samples),
            "num_event_rows": len(rows),
            "elapsed_seconds": elapsed,
            "outlier_thresholds": thresholds,
            "samples": sample_summaries,
            "skipped_samples": skipped_samples,
        },
    )

    print(f"Samples: {len(test_data)}")
    print(f"Profiled samples: {len(sample_summaries)}")
    print(f"Skipped samples: {len(skipped_samples)}")
    print(f"Event rows: {len(rows)}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Wrote teacher-forced activation profile to: {output_dir}")


if __name__ == "__main__":
    main()
