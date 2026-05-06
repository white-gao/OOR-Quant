#!/usr/bin/env python3
"""Export benchmark prompts as JSONL calibration data for llm-compressor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark import Benchmark
from benchmark.tasks.v1_0.registry import get_task_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export benchmark prompts to a JSONL file with a text column."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Model path used to load the tokenizer and chat template.",
    )
    parser.add_argument(
        "--data_dir",
        default="../data/onerec_data/benchmark_data",
        help="Benchmark data directory.",
    )
    parser.add_argument(
        "--task_type",
        default="ad",
        help="Benchmark task used for calibration prompts.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Benchmark split to export.",
    )
    parser.add_argument(
        "--sample_size",
        default="1000",
        help="Number of samples to export, or full.",
    )
    parser.add_argument(
        "--output",
        default="quantization_configs/calibration/ad_sample_1000.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--text_column",
        default="text",
        help="JSONL text column name.",
    )
    parser.add_argument(
        "--append_prompt_token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the task prompt_token, e.g. <|sid_begin|> for recommendation tasks.",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Use thinking chat template mode when loading benchmark prompts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    benchmark = Benchmark(
        model_path=args.model_path,
        task_types=[args.task_type],
        splits=[args.split],
        data_dir=args.data_dir,
        enable_thinking=args.enable_thinking,
    )
    data = benchmark.data_loader.load_data(
        task_name=args.task_type,
        split=args.split,
        sample_size=args.sample_size,
    )

    prompt_token = ""
    if args.append_prompt_token:
        prompt_token = get_task_config(args.task_type).get("prompt_token", "")

    with output_path.open("w", encoding="utf-8") as f:
        for sample in data.values():
            text = sample["prompt"] + prompt_token
            f.write(json.dumps({args.text_column: text}, ensure_ascii=False) + "\n")

    print(f"Exported {len(data)} calibration samples to {output_path}")
    if prompt_token:
        print(f"Appended prompt_token: {prompt_token}")


if __name__ == "__main__":
    main()
