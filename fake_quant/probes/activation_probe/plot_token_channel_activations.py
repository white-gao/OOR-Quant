#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
for path in (PROJECT_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.tasks.v1_0.registry import get_loader, get_task_config


DEFAULT_MODEL_PATH = "/home/guowei/OneRec-1.7B/"
DEFAULT_DATA_DIR = "data/onerec_data/benchmark-data"
DEFAULT_OUTPUT_DIR = "fake_quant/probes/activation_probe/activation_profiles/v1.0/token_channel_sample_0"
FALLBACK_MODEL_PATHS = [
    "/home/guowei/OneRec-1.7B/",
]
FALLBACK_DATA_DIRS = [
    "/zssd/home/yhhuang/Projects/OOR-Quant/data/onerec_data/benchmark-data",
    "/zssd/home/yhhuang/Projects/OOR-Quant/data/onerec_data/benchmark-data",
]


NODE_ORDER = [
    "attn_qkv_input",
    "attn_o_input",
    "ffn_gate_up_input",
    "ffn_down_input",
]

SID_ITEM_PATTERN = re.compile(
    r"<\|sid_begin\|><s_a_[^>]+><s_b_[^>]+><s_c_[^>]+><\|sid_end\|>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot token-channel activation maps for selected OneRec nodes.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--layers", default="0,4,8,12,16,20,24,27")
    parser.add_argument("--nodes", default=",".join(NODE_ORDER))
    parser.add_argument("--max_tokens", type=int, default=256, help="Left-truncate prompt to this many tokens before plotting; 0 disables token truncation.")
    parser.add_argument("--max_history_sid_items", type=int, default=0, help="Keep only this many complete history SID items; 0 keeps the history unchanged.")
    parser.add_argument("--history_sid_keep", default="last", choices=["first", "last"], help="Which history SID items to keep when compression is enabled.")
    parser.add_argument("--annotate_top_tokens", type=int, default=5, help="Annotate this many high-activation tokens on each plot; 0 disables peak labels.")
    parser.add_argument("--structure_label_mode", default="detailed", choices=["detailed", "compact"], help="Detailed labels mark each SID item; compact labels mark only broad prompt sections.")
    parser.add_argument("--channel_stride", type=int, default=4, help="Subsample channels for 3D surface plots.")
    parser.add_argument("--surface_max_tokens", type=int, default=128)
    parser.add_argument("--surface_max_channels", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_set(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def compress_history_sid_items(prompt: str, max_items: int, keep_policy: str) -> tuple[str, Dict[str, Any]]:
    matches = list(SID_ITEM_PATTERN.finditer(prompt))
    stats: Dict[str, Any] = {
        "original_history_sid_items": len(matches),
        "kept_history_sid_items": len(matches),
        "removed_history_sid_items": 0,
        "history_sid_keep": keep_policy,
        "original_prompt_chars": len(prompt),
        "compressed_prompt_chars": len(prompt),
    }
    if max_items <= 0 or len(matches) <= max_items:
        return prompt, stats

    if keep_policy == "last":
        keep_indices = set(range(len(matches) - max_items, len(matches)))
    else:
        keep_indices = set(range(max_items))

    parts: List[str] = []
    pos = 0
    for idx, match in enumerate(matches):
        parts.append(prompt[pos : match.start()])
        if idx in keep_indices:
            parts.append(match.group(0))
        pos = match.end()
    parts.append(prompt[pos:])

    compressed = "".join(parts)
    stats.update(
        {
            "kept_history_sid_items": max_items,
            "removed_history_sid_items": len(matches) - max_items,
            "compressed_prompt_chars": len(compressed),
        }
    )
    return compressed, stats


def _append_label(labels: Dict[int, str], token_index: int, label: str) -> None:
    existing = labels.get(token_index)
    labels[token_index] = label if not existing else f"{existing}; {label}"


def structural_token_labels(token_texts: List[str], mode: str = "detailed") -> Dict[int, str]:
    labels: Dict[int, str] = {}
    sid_begin_positions = [idx for idx, text in enumerate(token_texts) if text == "<|sid_begin|>"]
    sid_end_positions = [idx for idx, text in enumerate(token_texts) if text == "<|sid_end|>"]
    final_sid_begin = sid_begin_positions[-1] if sid_begin_positions and sid_begin_positions[-1] == len(token_texts) - 1 else None
    history_begins = [idx for idx in sid_begin_positions if idx != final_sid_begin]

    system_start = next((idx for idx, text in enumerate(token_texts[:-1]) if text == "<|im_start|>" and token_texts[idx + 1] == "system"), None)
    user_start = next((idx for idx, text in enumerate(token_texts[:-1]) if text == "<|im_start|>" and token_texts[idx + 1] == "user"), None)
    assistant_start = next((idx for idx, text in enumerate(token_texts[:-1]) if text == "<|im_start|>" and token_texts[idx + 1] == "assistant"), None)

    final_instruction_idx = None
    if sid_end_positions:
        final_instruction_idx = sid_end_positions[-1] + 1
        while final_instruction_idx < len(token_texts) and token_texts[final_instruction_idx].strip() == "":
            final_instruction_idx += 1

    if mode == "compact":
        if system_start is not None:
            _append_label(labels, system_start, "system start")
        if history_begins and sid_end_positions:
            _append_label(labels, history_begins[0], "history SID items start")
            _append_label(labels, sid_end_positions[-1], "history SID items end")
        if final_instruction_idx is not None and final_instruction_idx < len(token_texts):
            _append_label(labels, final_instruction_idx, "final instruction")
        if assistant_start is not None:
            _append_label(labels, assistant_start, "assistant start")
        if final_sid_begin is not None:
            _append_label(labels, final_sid_begin, "decode <|sid_begin|>")
        return labels

    for idx, text in enumerate(token_texts):
        if text != "<|im_start|>":
            continue
        role = token_texts[idx + 1] if idx + 1 < len(token_texts) else ""
        if role in {"system", "user", "assistant"}:
            _append_label(labels, idx, f"{role} start")
        else:
            _append_label(labels, idx, "<|im_start|>")

    for idx, text in enumerate(token_texts):
        if text == "<|im_end|>":
            _append_label(labels, idx, "turn end")

    for ordinal, idx in enumerate(history_begins, 1):
        _append_label(labels, idx, f"history SID{ordinal} begin")
    for ordinal, idx in enumerate(sid_end_positions, 1):
        if ordinal == 1 or ordinal == len(sid_end_positions):
            _append_label(labels, idx, f"history SID{ordinal} end")
    if final_sid_begin is not None:
        _append_label(labels, final_sid_begin, "decode <|sid_begin|>")

    if user_start is not None:
        header_idx = min(user_start + 3, len(token_texts) - 1)
        _append_label(labels, header_idx, "history prompt header")
    if final_instruction_idx is not None and final_instruction_idx < len(token_texts):
        _append_label(labels, final_instruction_idx, "final instruction")
    if assistant_start is not None:
        think_idx = next((idx for idx in range(assistant_start, len(token_texts)) if token_texts[idx] == "<think>"), None)
        if think_idx is not None:
            _append_label(labels, think_idx, "assistant think block")
    return labels


def add_top_activation_labels(labels: Dict[int, str], matrix: np.ndarray, token_texts: List[str], top_k: int) -> Dict[int, str]:
    merged = dict(labels)
    if top_k <= 0 or matrix.size == 0:
        return merged
    row_max = matrix.max(axis=1)
    top_indices = np.argsort(row_max)[-top_k:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        token_text = token_texts[int(idx)] if int(idx) < len(token_texts) else ""
        token_text = token_text.replace("\n", "\\n")
        _append_label(merged, int(idx), f"top{rank} max {token_text}")
    return merged


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_model_path(model_path: str) -> str:
    path = Path(model_path)
    if path.exists():
        return str(path)
    if model_path == DEFAULT_MODEL_PATH:
        for fallback in FALLBACK_MODEL_PATHS:
            if Path(fallback).exists():
                print(f"Default model path not found: {DEFAULT_MODEL_PATH}")
                print(f"Using fallback model path: {fallback}")
                return fallback
    raise FileNotFoundError(f"Model path not found: {model_path}")


def resolve_data_dir(data_dir: str) -> str:
    path = Path(data_dir)
    if (path / "ad" / "ad_test.parquet").exists():
        return str(path)
    if data_dir == DEFAULT_DATA_DIR:
        for fallback in FALLBACK_DATA_DIRS:
            if (Path(fallback) / "ad" / "ad_test.parquet").exists():
                print(f"Default data dir not found: {DEFAULT_DATA_DIR}")
                print(f"Using fallback data dir: {fallback}")
                return fallback
    raise FileNotFoundError(f"AD benchmark data not found under: {data_dir}")


def safe_token_text(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    return text.replace("\n", "\\n").replace("\r", "\\r")


def token_group(token_text: str) -> str:
    if token_text in {"<|sid_begin|>", "<|sid_end|>"}:
        return "sid_boundary"
    if token_text.startswith("<s_a_") or token_text.startswith("<s_b_") or token_text.startswith("<s_c_"):
        return "sid_code"
    if token_text.startswith("<|im_") or token_text in {"<think>", "</think>"}:
        return "chat_special"
    return "prompt_text"


def load_sample(tokenizer: Any, data_dir: str, sample_index: int) -> tuple[str, Dict[str, Any]]:
    loader = get_loader("ad", data_dir=data_dir, tokenizer=tokenizer, enable_thinking=False)
    data = loader.load_data(split="test", sample_size=sample_index + 1)
    keys = list(data.keys())
    if sample_index >= len(keys):
        raise IndexError(f"sample_index={sample_index} but only loaded {len(keys)} samples")
    sample_id = keys[sample_index]
    return sample_id, data[sample_id]


def effective_generation_prompt(prompt: str) -> str:
    prompt_token = get_task_config("ad").get("generation_config", {}).get("prompt_token", "<|sid_begin|>")
    if prompt_token and not prompt.endswith(prompt_token):
        return prompt + prompt_token
    return prompt


def encode_prompt(tokenizer: Any, prompt: str, max_tokens: int) -> torch.Tensor:
    prompt = effective_generation_prompt(prompt)
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if max_tokens and input_ids.shape[1] > max_tokens:
        input_ids = input_ids[:, -max_tokens:]
    return input_ids


@dataclass
class CaptureContext:
    token_ids: List[int]
    token_texts: List[str]
    token_groups: List[str]


class TokenChannelCapture:
    def __init__(self, *, layers: set[int], nodes: set[str], context: CaptureContext) -> None:
        self.layers = layers
        self.nodes = nodes
        self.context = context
        self.handles: List[Any] = []
        self.records: Dict[tuple[int, str], torch.Tensor] = {}

    def add_output_hook(self, module: torch.nn.Module, layer: int, node: str) -> None:
        if layer not in self.layers or node not in self.nodes:
            return

        def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            self._store(layer, node, tensor)

        self.handles.append(module.register_forward_hook(hook))

    def add_input_hook(self, module: torch.nn.Module, layer: int, node: str) -> None:
        if layer not in self.layers or node not in self.nodes:
            return

        def hook(_module: torch.nn.Module, inputs: Any, _output: Any) -> None:
            if not inputs:
                return
            self._store(layer, node, inputs[0])

        self.handles.append(module.register_forward_hook(hook))

    def _store(self, layer: int, node: str, tensor: Any) -> None:
        if not torch.is_tensor(tensor):
            return
        if tensor.dim() >= 3:
            tensor = tensor[0]
        if tensor.dim() != 2:
            return
        seq_len = min(tensor.shape[0], len(self.context.token_ids))
        self.records[(layer, node)] = tensor[:seq_len].detach().float().abs().cpu()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def register_capture_hooks(model: torch.nn.Module, capture: TokenChannelCapture) -> None:
    for layer_idx, layer in enumerate(model.model.layers):
        capture.add_output_hook(layer.input_layernorm, layer_idx, "attn_qkv_input")
        capture.add_input_hook(layer.self_attn.o_proj, layer_idx, "attn_o_input")
        capture.add_output_hook(layer.post_attention_layernorm, layer_idx, "ffn_gate_up_input")
        capture.add_input_hook(layer.mlp.down_proj, layer_idx, "ffn_down_input")


def plot_heatmap(
    path: Path,
    matrix: np.ndarray,
    title: str,
    token_groups: List[str],
    important_tokens: Optional[Dict[int, str]] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("channel")
    ax.set_ylabel("token position")

    group_colors = {
        "chat_special": "#d62728",
        "prompt_text": "#1f77b4",
        "sid_boundary": "#ff7f0e",
        "sid_code": "#2ca02c",
    }
    for idx, group in enumerate(token_groups):
        color = group_colors.get(group)
        if color:
            ax.plot([-0.5, -0.5], [idx - 0.5, idx + 0.5], color=color, linewidth=3, clip_on=False)

    important_tokens = important_tokens or {}
    for idx, label in sorted(important_tokens.items()):
        if idx < 0 or idx >= matrix.shape[0]:
            continue
        ax.axhline(idx, color="#00e5ff", linewidth=0.6, alpha=0.8)
        ax.text(1.01, idx, label, transform=ax.get_yaxis_transform(), va="center", fontsize=6, color="#005f73")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("|activation|")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_surface(
    path: Path,
    matrix: np.ndarray,
    title: str,
    channel_stride: int,
    important_tokens: Optional[Dict[int, str]] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    path.parent.mkdir(parents=True, exist_ok=True)
    values = matrix
    token_idx = np.arange(values.shape[0])
    channel_idx = np.arange(values.shape[1]) * channel_stride
    x, y = np.meshgrid(channel_idx, token_idx)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x, y, values, cmap="viridis", linewidth=0, antialiased=True, rstride=1, cstride=1)
    important_tokens = important_tokens or {}
    y_ticks: List[int] = []
    y_tick_labels: List[str] = []
    for idx, label in sorted(important_tokens.items()):
        if idx < 0 or idx >= values.shape[0]:
            continue
        y_ticks.append(idx)
        y_tick_labels.append(f"{idx}:{label}")
    if y_ticks:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels, fontsize=6)
    ax.set_title(title)
    ax.set_xlabel("channel")
    ax.set_ylabel("token position")
    ax.set_zlabel("|activation|")
    fig.colorbar(surf, shrink=0.55, aspect=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_token_metadata(path: Path, context: CaptureContext, important_labels: Optional[Dict[int, str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    important_labels = important_labels or {}
    rows = [
        {
            "token_index": idx,
            "token_id": token_id,
            "token_text": text,
            "token_group": group,
            "important_label": important_labels.get(idx),
        }
        for idx, (token_id, text, group) in enumerate(
            zip(context.token_ids, context.token_texts, context.token_groups)
        )
    ]
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    layers = set(parse_int_list(args.layers))
    nodes = parse_str_set(args.nodes)
    unknown_nodes = nodes - set(NODE_ORDER)
    if unknown_nodes:
        raise ValueError(f"Unknown nodes: {sorted(unknown_nodes)}")

    output_dir = Path(args.output_dir)
    model_path = resolve_model_path(args.model_path)
    data_dir = resolve_data_dir(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sample_id, sample = load_sample(tokenizer, data_dir, args.sample_index)
    prompt, prompt_compression = compress_history_sid_items(
        sample["prompt"],
        max_items=args.max_history_sid_items,
        keep_policy=args.history_sid_keep,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_prompt = effective_generation_prompt(prompt)
    (output_dir / "compressed_prompt.txt").write_text(effective_prompt, encoding="utf-8")
    input_ids = encode_prompt(tokenizer, prompt, args.max_tokens)
    token_ids = input_ids[0].tolist()
    token_texts = [safe_token_text(tokenizer, token_id) for token_id in token_ids]
    token_groups = [token_group(text) for text in token_texts]
    structural_labels = structural_token_labels(token_texts, mode=args.structure_label_mode)
    context = CaptureContext(token_ids=token_ids, token_texts=token_texts, token_groups=token_groups)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype(args.dtype),
        device_map=None,
    ).to(args.device)
    model.eval()

    capture = TokenChannelCapture(layers=layers, nodes=nodes, context=context)
    register_capture_hooks(model, capture)
    try:
        with torch.inference_mode():
            _ = model(input_ids=input_ids.to(args.device), use_cache=False)
    finally:
        capture.close()

    write_token_metadata(output_dir / "token_metadata.json", context, structural_labels)
    summary: Dict[str, Any] = {
        "sample_id": sample_id,
        "sample_index": args.sample_index,
        "prompt_tokens": len(token_ids),
        "max_tokens": args.max_tokens,
        "prompt_compression": prompt_compression,
        "layers": sorted(layers),
        "nodes": sorted(nodes),
        "structure_label_mode": args.structure_label_mode,
        "plots": [],
    }

    for (layer, node), tensor in sorted(capture.records.items()):
        matrix = tensor.numpy()
        important_labels = add_top_activation_labels(
            structural_labels,
            matrix,
            context.token_texts,
            args.annotate_top_tokens,
        )
        heatmap_path = output_dir / "heatmaps" / f"layer_{layer:02d}__{node}.png"
        plot_heatmap(
            heatmap_path,
            matrix,
            title=f"sample {sample_id} | layer {layer} | {node} | token-channel heatmap",
            token_groups=context.token_groups[: matrix.shape[0]],
            important_tokens=important_labels,
        )

        surface = matrix
        surface_offset = 0
        if args.surface_max_tokens and surface.shape[0] > args.surface_max_tokens:
            surface_offset = surface.shape[0] - args.surface_max_tokens
            surface = surface[-args.surface_max_tokens :, :]
        stride = max(args.channel_stride, 1)
        surface = surface[:, ::stride]
        if args.surface_max_channels and surface.shape[1] > args.surface_max_channels:
            surface = surface[:, : args.surface_max_channels]
        surface_labels = {
            idx - surface_offset: label
            for idx, label in important_labels.items()
            if surface_offset <= idx < surface_offset + surface.shape[0]
        }
        surface_path = output_dir / "surfaces" / f"layer_{layer:02d}__{node}_surface.png"
        plot_surface(
            surface_path,
            surface,
            title=f"sample {sample_id} | layer {layer} | {node} | 3D surface",
            channel_stride=stride,
            important_tokens=surface_labels,
        )

        row_max = matrix.max(axis=1)
        summary["plots"].append(
            {
                "layer": layer,
                "node": node,
                "shape": list(matrix.shape),
                "heatmap": str(heatmap_path),
                "surface": str(surface_path),
                "absmax": float(matrix.max()),
                "p999": float(np.quantile(matrix.reshape(-1), 0.999)),
                "important_tokens": [
                    {
                        "token_index": int(idx),
                        "token_text": context.token_texts[int(idx)] if int(idx) < len(context.token_texts) else "",
                        "token_group": context.token_groups[int(idx)] if int(idx) < len(context.token_groups) else "",
                        "label": label,
                        "row_absmax": float(row_max[int(idx)]) if int(idx) < len(row_max) else None,
                    }
                    for idx, label in sorted(important_labels.items())
                    if 0 <= int(idx) < matrix.shape[0]
                ],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Sample ID: {sample_id}")
    print(f"Prompt tokens: {len(token_ids)}")
    print(f"Captured nodes: {len(capture.records)}")
    print(f"Wrote token-channel plots to: {output_dir}")


if __name__ == "__main__":
    main()
