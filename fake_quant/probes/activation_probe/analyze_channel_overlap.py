#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
for path in (PROJECT_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.tasks.v1_0.registry import get_loader, get_task_config


DEFAULT_MODEL_PATH = "/home/guowei/OneRec-1.7B"
DEFAULT_DATA_DIR = "data/onerec_data/benchmark-data"
DEFAULT_OUTPUT_DIR = "fake_quant/probes/activation_probe/activation_profiles/v1.0/channel_overlap_sample_0"
FALLBACK_MODEL_PATHS = [
    "/zssd/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B",
    "/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B",
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
METRICS = ["mean_abs", "p99_abs", "max_abs"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze whether outlier channels are fixed across layers.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--layers", default="all")
    parser.add_argument("--nodes", default=",".join(NODE_ORDER))
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def parse_int_list(value: str, num_layers: int) -> List[int]:
    if value == "all":
        return list(range(num_layers))
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


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


def load_sample(tokenizer: Any, data_dir: str, sample_index: int) -> tuple[str, Dict[str, Any]]:
    loader = get_loader("ad", data_dir=data_dir, tokenizer=tokenizer, enable_thinking=False)
    data = loader.load_data(split="test", sample_size=sample_index + 1)
    keys = list(data.keys())
    if sample_index >= len(keys):
        raise IndexError(f"sample_index={sample_index} but only loaded {len(keys)} samples")
    sample_id = keys[sample_index]
    return sample_id, data[sample_id]


def encode_prompt(tokenizer: Any, prompt: str, max_tokens: int) -> torch.Tensor:
    prompt_token = get_task_config("ad").get("generation_config", {}).get("prompt_token", "<|sid_begin|>")
    if prompt_token and not prompt.endswith(prompt_token):
        prompt = prompt + prompt_token
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if max_tokens and input_ids.shape[1] > max_tokens:
        input_ids = input_ids[:, -max_tokens:]
    return input_ids


@dataclass
class ChannelScoreRecord:
    layer: int
    node: str
    scores: Dict[str, torch.Tensor]


class ChannelScoreCapture:
    def __init__(self, *, layers: set[int], nodes: set[str]) -> None:
        self.layers = layers
        self.nodes = nodes
        self.handles: List[Any] = []
        self.records: Dict[tuple[int, str], ChannelScoreRecord] = {}

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
            if inputs:
                self._store(layer, node, inputs[0])

        self.handles.append(module.register_forward_hook(hook))

    def _store(self, layer: int, node: str, tensor: Any) -> None:
        if not torch.is_tensor(tensor):
            return
        if tensor.dim() >= 3:
            tensor = tensor[0]
        if tensor.dim() != 2:
            return
        values = tensor.detach().float().abs().cpu()
        self.records[(layer, node)] = ChannelScoreRecord(
            layer=layer,
            node=node,
            scores={
                "mean_abs": values.mean(dim=0),
                "p99_abs": torch.quantile(values, 0.99, dim=0),
                "max_abs": values.max(dim=0).values,
            },
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def register_capture_hooks(model: torch.nn.Module, capture: ChannelScoreCapture) -> None:
    for layer_idx, layer in enumerate(model.model.layers):
        capture.add_output_hook(layer.input_layernorm, layer_idx, "attn_qkv_input")
        capture.add_input_hook(layer.self_attn.o_proj, layer_idx, "attn_o_input")
        capture.add_output_hook(layer.post_attention_layernorm, layer_idx, "ffn_gate_up_input")
        capture.add_input_hook(layer.mlp.down_proj, layer_idx, "ffn_down_input")


def topk_indices(scores: torch.Tensor, topk: int) -> List[int]:
    k = min(topk, scores.numel())
    return torch.topk(scores, k=k, largest=True).indices.tolist()


def overlap_matrix(top_channels_by_layer: Dict[int, List[int]], layers: List[int]) -> np.ndarray:
    mat = np.zeros((len(layers), len(layers)), dtype=np.float32)
    sets = {layer: set(top_channels_by_layer[layer]) for layer in layers}
    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            denom = max(len(sets[li]), 1)
            mat[i, j] = len(sets[li] & sets[lj]) / denom
    return mat


def channel_frequency(top_channels_by_layer: Dict[int, List[int]]) -> Dict[int, int]:
    freq: Dict[int, int] = {}
    for channels in top_channels_by_layer.values():
        for channel in channels:
            freq[channel] = freq.get(channel, 0) + 1
    return freq


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_overlap(path: Path, matrix: np.ndarray, layers: List[int], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("layer")
    ax.set_ylabel("layer")
    ax.set_xticks(range(len(layers)), layers, rotation=90)
    ax.set_yticks(range(len(layers)), layers)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("topK overlap")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_frequency(path: Path, freq: Dict[int, int], num_layers: int, title: str, limit: int = 64) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(freq.items(), key=lambda item: (-item[1], item[0]))[:limit]
    channels = [item[0] for item in items]
    counts = [item[1] for item in items]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(range(len(channels)), counts)
    ax.set_title(title)
    ax.set_xlabel("channel rank by frequency")
    ax.set_ylabel(f"#layers in topK / {num_layers}")
    ax.set_xticks(range(len(channels)), [str(ch) for ch in channels], rotation=90, fontsize=7)
    ax.set_ylim(0, num_layers)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_union_scores(
    path: Path,
    records: Dict[int, ChannelScoreRecord],
    layers: List[int],
    metric: str,
    channels: List[int],
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.zeros((len(layers), len(channels)), dtype=np.float32)
    for i, layer in enumerate(layers):
        scores = records[layer].scores[metric].numpy()
        for j, channel in enumerate(channels):
            if channel < len(scores):
                data[i, j] = scores[channel]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(10, len(channels) * 0.25), 6))
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("top channel union")
    ax.set_ylabel("layer")
    ax.set_xticks(range(len(channels)), [str(ch) for ch in channels], rotation=90, fontsize=7)
    ax.set_yticks(range(len(layers)), layers)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze_node_metric(
    *,
    node: str,
    metric: str,
    records: Dict[int, ChannelScoreRecord],
    layers: List[int],
    topk: int,
    output_dir: Path,
) -> Dict[str, Any]:
    top_by_layer = {
        layer: topk_indices(records[layer].scores[metric], topk)
        for layer in layers
        if layer in records
    }
    valid_layers = [layer for layer in layers if layer in top_by_layer]
    if not valid_layers:
        return {}

    matrix = overlap_matrix(top_by_layer, valid_layers)
    freq = channel_frequency(top_by_layer)
    channel_dim = int(records[valid_layers[0]].scores[metric].numel())
    random_expected = topk / channel_dim

    safe = f"{node}__{metric}"
    plot_overlap(
        output_dir / "plots" / f"{safe}_overlap.png",
        matrix,
        valid_layers,
        title=f"{node} / {metric} top{topk} channel overlap",
    )
    plot_frequency(
        output_dir / "plots" / f"{safe}_frequency.png",
        freq,
        len(valid_layers),
        title=f"{node} / {metric} top{topk} channel frequency",
    )
    union_channels = sorted(freq, key=lambda channel: (-freq[channel], channel))[: min(96, len(freq))]
    plot_union_scores(
        output_dir / "plots" / f"{safe}_union_scores.png",
        records,
        valid_layers,
        metric,
        union_channels,
        title=f"{node} / {metric} scores on frequent top channels",
    )

    top_rows: List[Dict[str, Any]] = []
    for layer in valid_layers:
        scores = records[layer].scores[metric]
        values, indices = torch.topk(scores, k=min(topk, scores.numel()), largest=True)
        for rank, (channel, value) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
            top_rows.append(
                {
                    "node": node,
                    "metric": metric,
                    "layer": layer,
                    "rank": rank,
                    "channel": channel,
                    "score": value,
                }
            )
    write_csv(output_dir / "csv" / f"{safe}_top_channels.csv", top_rows)

    freq_rows = [
        {
            "node": node,
            "metric": metric,
            "channel": channel,
            "frequency": count,
            "frequency_ratio": count / len(valid_layers),
        }
        for channel, count in sorted(freq.items(), key=lambda item: (-item[1], item[0]))
    ]
    write_csv(output_dir / "csv" / f"{safe}_channel_frequency.csv", freq_rows)

    off_diag = matrix[~np.eye(matrix.shape[0], dtype=bool)] if matrix.shape[0] > 1 else np.array([])
    return {
        "node": node,
        "metric": metric,
        "channel_dim": channel_dim,
        "topk": topk,
        "random_expected_overlap": random_expected,
        "mean_offdiag_overlap": float(off_diag.mean()) if off_diag.size else None,
        "max_offdiag_overlap": float(off_diag.max()) if off_diag.size else None,
        "max_channel_frequency": max(freq.values()) if freq else 0,
        "max_channel_frequency_ratio": (max(freq.values()) / len(valid_layers)) if freq else 0.0,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    output_dir = Path(args.output_dir)
    model_path = resolve_model_path(args.model_path)
    data_dir = resolve_data_dir(args.data_dir)
    nodes = parse_str_list(args.nodes)
    unknown_nodes = set(nodes) - set(NODE_ORDER)
    if unknown_nodes:
        raise ValueError(f"Unknown nodes: {sorted(unknown_nodes)}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sample_id, sample = load_sample(tokenizer, data_dir, args.sample_index)
    input_ids = encode_prompt(tokenizer, sample["prompt"], args.max_tokens)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype(args.dtype),
        device_map=None,
    ).to(args.device)
    model.eval()

    layers = parse_int_list(args.layers, len(model.model.layers))
    capture = ChannelScoreCapture(layers=set(layers), nodes=set(nodes))
    register_capture_hooks(model, capture)
    try:
        with torch.inference_mode():
            _ = model(input_ids=input_ids.to(args.device), use_cache=False)
    finally:
        capture.close()

    records_by_node: Dict[str, Dict[int, ChannelScoreRecord]] = {node: {} for node in nodes}
    for (layer, node), record in capture.records.items():
        records_by_node.setdefault(node, {})[layer] = record

    summaries = []
    for node in nodes:
        for metric in METRICS:
            summary = analyze_node_metric(
                node=node,
                metric=metric,
                records=records_by_node.get(node, {}),
                layers=layers,
                topk=args.topk,
                output_dir=output_dir,
            )
            if summary:
                summaries.append(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "sample_index": args.sample_index,
                "prompt_tokens": int(input_ids.shape[1]),
                "layers": layers,
                "nodes": nodes,
                "topk": args.topk,
                "metrics": METRICS,
                "summaries": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Sample ID: {sample_id}")
    print(f"Prompt tokens: {int(input_ids.shape[1])}")
    print(f"Captured records: {len(capture.records)}")
    print(f"Wrote channel overlap analysis to: {output_dir}")


if __name__ == "__main__":
    main()
