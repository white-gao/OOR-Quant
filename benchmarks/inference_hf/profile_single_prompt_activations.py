#!/usr/bin/env python3
"""
Profile per-token activations for one formatted prompt with Transformers.

This is intended for debugging/profiling, not benchmark inference. It uses HF
Transformers because vLLM is optimized for serving and does not expose internal
per-layer activations cleanly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = (
    "/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/"
    "snapshots/OneRec-1.7B"
)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLE_PATH = SCRIPT_DIR / "sample" / "sample_1.json"
DEFAULT_OUTPUT_PREFIX = SCRIPT_DIR / "results" / "activation_profiles" / "sample_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile token activations for a single prompt.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sample_path", default=str(DEFAULT_SAMPLE_PATH))
    parser.add_argument("--output_prefix", default=str(DEFAULT_OUTPUT_PREFIX))
    parser.add_argument(
        "--append_prompt_token",
        default="<|sid_begin|>",
        help="Token appended before SID prediction. Use empty string to disable.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for running the forward pass.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=0,
        help="Optional left truncation limit. 0 means keep the full prompt.",
    )
    parser.add_argument(
        "--token_detail_window",
        type=int,
        default=64,
        help="Write detailed rows for the last N tokens; all tokens are still included in aggregate rows.",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_prompt(sample_path: Path, append_prompt_token: str) -> str:
    data = json.loads(sample_path.read_text(encoding="utf-8"))
    prompt = data["prompt"] if isinstance(data, dict) else str(data)
    if append_prompt_token and not prompt.endswith(append_prompt_token):
        prompt += append_prompt_token
    return prompt


def safe_token_text(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    return text.replace("\n", "\\n").replace("\r", "\\r")


def token_type(token_text: str) -> str:
    if token_text in {"<|sid_begin|>", "<|sid_end|>"}:
        return "sid_boundary"
    if token_text.startswith("<s_a_"):
        return "sid_a"
    if token_text.startswith("<s_b_"):
        return "sid_b"
    if token_text.startswith("<s_c_"):
        return "sid_c"
    if token_text.startswith("<|im_") or token_text in {"<think>", "</think>"}:
        return "chat_special"
    return "text"


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    values = x.detach().float().abs().flatten()
    if values.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "absmax": 0.0, "p99": 0.0, "p999": 0.0}
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "absmax": float(values.max().item()),
        "p99": float(torch.quantile(values, 0.99).item()),
        "p999": float(torch.quantile(values, 0.999).item()),
    }


def finite_or_blank(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


class ActivationCollector:
    def __init__(self, token_ids: List[int], token_texts: List[str], token_types: List[str]) -> None:
        self.token_ids = token_ids
        self.token_texts = token_texts
        self.token_types = token_types
        self.rows: List[Dict[str, Any]] = []
        self.handles: List[Any] = []

    def add_hook(self, module: torch.nn.Module, layer: int, module_name: str) -> None:
        def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(tensor):
                return
            # Expected shape: [batch, seq, hidden/out_features].
            if tensor.dim() < 3:
                return
            tensor = tensor[0]
            seq_len = min(tensor.shape[0], len(self.token_ids))
            for idx in range(seq_len):
                stats = tensor_stats(tensor[idx])
                self.rows.append(
                    {
                        "layer": layer,
                        "module": module_name,
                        "token_index": idx,
                        "token_id": self.token_ids[idx],
                        "token_text": self.token_texts[idx],
                        "token_type": self.token_types[idx],
                        **stats,
                    }
                )

        self.handles.append(module.register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def aggregate_rows(rows: Iterable[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    buckets: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        bucket_key = tuple(row[key] for key in keys)
        buckets.setdefault(bucket_key, []).append(row)

    out: List[Dict[str, Any]] = []
    for bucket_key, bucket_rows in sorted(buckets.items()):
        item = {key: value for key, value in zip(keys, bucket_key)}
        item["count"] = len(bucket_rows)
        for stat in ["mean", "std", "absmax", "p99", "p999"]:
            values = [float(row[stat]) for row in bucket_rows]
            item[f"{stat}_avg"] = sum(values) / len(values)
            item[f"{stat}_max"] = max(values)
        out.append(item)
    return out


def last_token_rows(rows: Iterable[Dict[str, Any]], last_token_index: int) -> List[Dict[str, Any]]:
    return [row for row in rows if int(row["token_index"]) == last_token_index]


def late_layer_rows(rows: Iterable[Dict[str, Any]], first_late_layer: int) -> List[Dict[str, Any]]:
    return [row for row in rows if int(row["layer"]) >= first_late_layer]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: finite_or_blank(value) for key, value in row.items()})


def main() -> None:
    args = parse_args()
    sample_path = Path(args.sample_path)
    output_prefix = Path(args.output_prefix)

    prompt = load_prompt(sample_path, args.append_prompt_token)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if args.max_tokens and input_ids.shape[1] > args.max_tokens:
        input_ids = input_ids[:, -args.max_tokens :]

    token_ids = input_ids[0].tolist()
    token_texts = [safe_token_text(tokenizer, token_id) for token_id in token_ids]
    token_types = [token_type(text) for text in token_texts]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype(args.dtype),
        device_map=None,
    ).to(device)
    model.eval()

    collector = ActivationCollector(token_ids, token_texts, token_types)
    num_layers = len(model.model.layers)
    for layer_idx, layer in enumerate(model.model.layers):
        # Normalized inputs are the most relevant activations for future W8A8
        # experiments because they are the actual inputs to attention/MLP linear
        # projections.
        collector.add_hook(layer.input_layernorm, layer_idx, "attn_input_norm")
        collector.add_hook(layer.post_attention_layernorm, layer_idx, "mlp_input_norm")
        collector.add_hook(layer, layer_idx, "residual_block_output")
        collector.add_hook(layer.mlp.gate_proj, layer_idx, "mlp.gate_proj")
        collector.add_hook(layer.mlp.up_proj, layer_idx, "mlp.up_proj")
        collector.add_hook(layer.mlp.down_proj, layer_idx, "mlp.down_proj")
        collector.add_hook(layer.self_attn.o_proj, layer_idx, "self_attn.o_proj")
    collector.add_hook(model.model.norm, num_layers, "final_norm")

    with torch.inference_mode():
        _ = model(input_ids=input_ids.to(device), use_cache=False)
    collector.close()

    detail_window = max(args.token_detail_window, 0)
    if detail_window:
        min_idx = max(0, len(token_ids) - detail_window)
        detail_rows = [row for row in collector.rows if int(row["token_index"]) >= min_idx]
    else:
        detail_rows = collector.rows

    token_rows = [
        {
            "token_index": idx,
            "token_id": token_id,
            "token_text": token_texts[idx],
            "token_type": token_types[idx],
        }
        for idx, token_id in enumerate(token_ids)
    ]

    write_csv(output_prefix.with_name(output_prefix.name + "_tokens.csv"), token_rows)
    write_csv(output_prefix.with_name(output_prefix.name + "_detail.csv"), detail_rows)
    write_csv(
        output_prefix.with_name(output_prefix.name + "_last_token_by_layer_module.csv"),
        last_token_rows(collector.rows, len(token_ids) - 1),
    )
    write_csv(
        output_prefix.with_name(output_prefix.name + "_by_module_token_type.csv"),
        aggregate_rows(collector.rows, ["module", "token_type"]),
    )
    write_csv(
        output_prefix.with_name(output_prefix.name + "_by_layer_module.csv"),
        aggregate_rows(collector.rows, ["layer", "module"]),
    )
    write_csv(
        output_prefix.with_name(output_prefix.name + "_by_layer_module_token_type.csv"),
        aggregate_rows(collector.rows, ["layer", "module", "token_type"]),
    )
    write_csv(
        output_prefix.with_name(output_prefix.name + "_late_layers_by_module_token_type.csv"),
        aggregate_rows(late_layer_rows(collector.rows, max(0, num_layers - 8)), ["module", "token_type"]),
    )

    print(f"Prompt tokens: {len(token_ids)}")
    print(f"Activation rows: {len(collector.rows)}")
    print(f"Wrote: {output_prefix.with_name(output_prefix.name + '_tokens.csv')}")
    print(f"Wrote: {output_prefix.with_name(output_prefix.name + '_detail.csv')}")
    print(f"Wrote: {output_prefix.with_name(output_prefix.name + '_last_token_by_layer_module.csv')}")
    print(f"Wrote: {output_prefix.with_name(output_prefix.name + '_by_module_token_type.csv')}")
    print(f"Wrote: {output_prefix.with_name(output_prefix.name + '_by_layer_module.csv')}")
    print(f"Wrote: {output_prefix.with_name(output_prefix.name + '_by_layer_module_token_type.csv')}")
    print(f"Wrote: {output_prefix.with_name(output_prefix.name + '_late_layers_by_module_token_type.csv')}")


if __name__ == "__main__":
    main()
