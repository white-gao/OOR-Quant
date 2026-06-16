#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
for path in (PROJECT_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.tasks.v1_0.registry import get_loader, get_task_config  # noqa: E402


DEFAULT_MODEL_PATH = "/home/guowei/OneRec-1.7B/"
DEFAULT_DATA_DIR = "data/onerec_data/benchmark-data-calib1024"
DEFAULT_OUTPUT_DIR = "fake_quant_learnable/results/analysis/token_modality_gap/mquant_style"
DEFAULT_LAYERS = "0,8,16,24,27"
DEFAULT_NODES = "attn_qkv_input,attn_o_input,ffn_gate_up_input,ffn_down_input"
SID_ITEM_PATTERN = re.compile(
    r"<\|sid_begin\|><s_a_[^>]+><s_b_[^>]+><s_c_[^>]+><\|sid_end\|>"
)
SID_CODE_PATTERN = re.compile(r"^<s_[abc]_[^>]+>$")
CHAT_SPECIAL_TOKENS = {"<|im_start|>", "<|im_end|>", "<think>", "</think>"}
SID_BOUNDARY_TOKENS = {"<|sid_begin|>", "<|sid_end|>"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OneRec text-token and SID-token activation distributions, MQuant Fig.1(b)-style."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="test", choices=["calib", "test"])
    parser.add_argument("--sample_size", type=int, default=4)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--nodes", default=DEFAULT_NODES)
    parser.add_argument("--max_history_sid_items", type=int, default=10)
    parser.add_argument("--history_sid_keep", default="last", choices=["first", "last"])
    parser.add_argument("--max_tokens", type=int, default=0, help="Left-truncate encoded prompt to this many tokens; 0 disables truncation.")
    parser.add_argument("--hist_bins", type=int, default=160)
    parser.add_argument("--plot_max_values", type=int, default=300_000, help="Max sampled values per modality in each subplot.")
    parser.add_argument("--capture_max_values_per_modality", type=int, default=120_000, help="Max activation values kept per sample/layer/node/modality for approximate distribution stats.")
    parser.add_argument("--x_clip_quantile", type=float, default=0.9995)
    parser.add_argument("--skip_abs_tail", action="store_true", help="Only draw signed histograms and skip absolute-value survival plots.")
    parser.add_argument("--sid_boundary_as_sid", action="store_true", help="Treat <|sid_begin|> and <|sid_end|> as SID tokens instead of excluding them.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def effective_generation_prompt(prompt: str) -> str:
    prompt_token = get_task_config("ad").get("generation_config", {}).get("prompt_token", "<|sid_begin|>")
    if prompt_token and not prompt.endswith(prompt_token):
        return prompt + prompt_token
    return prompt


def compress_history_sid_items(prompt: str, max_items: int, keep_policy: str) -> tuple[str, dict[str, Any]]:
    matches = list(SID_ITEM_PATTERN.finditer(prompt))
    stats: dict[str, Any] = {
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

    parts: list[str] = []
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


def safe_token_text(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    return text.replace("\n", "\\n").replace("\r", "\\r")


def token_modality(token_text: str, *, sid_boundary_as_sid: bool = False) -> str | None:
    """Return the two groups compared in this experiment: text vs SID tokens."""
    if token_text in CHAT_SPECIAL_TOKENS:
        return None
    if token_text in SID_BOUNDARY_TOKENS:
        return "sid_code" if sid_boundary_as_sid else None
    if SID_CODE_PATTERN.match(token_text):
        return "sid_code"
    if token_text.startswith("<|"):
        return None
    return "text"


def encode_prompt(tokenizer: Any, prompt: str, max_tokens: int) -> torch.Tensor:
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if max_tokens and input_ids.shape[1] > max_tokens:
        input_ids = input_ids[:, -max_tokens:]
    return input_ids


def load_ad_samples(tokenizer: Any, data_dir: str, split: str, sample_size: int, sample_offset: int) -> list[tuple[str, Mapping[str, Any]]]:
    if sample_size <= 0:
        raise ValueError("--sample_size must be positive")
    if sample_offset < 0:
        raise ValueError("--sample_offset must be non-negative")
    loader = get_loader("ad", data_dir=data_dir, tokenizer=tokenizer, enable_thinking=False)
    loaded = loader.load_data(split=split, sample_size=sample_size + sample_offset)
    items = list(loaded.items())[sample_offset : sample_offset + sample_size]
    if len(items) < sample_size:
        raise ValueError(f"Requested {sample_size} samples after offset {sample_offset}, got {len(items)}")
    return items


@dataclass(frozen=True)
class TokenContext:
    sample_id: str
    token_ids: list[int]
    token_texts: list[str]
    modalities: list[str | None]


class ModalityActivationCapture:
    def __init__(self, *, layers: set[int], nodes: set[str], max_values_per_modality: int, seed: int) -> None:
        self.layers = layers
        self.nodes = nodes
        self.max_values_per_modality = int(max_values_per_modality)
        self.seed = int(seed)
        self.context: TokenContext | None = None
        self.handles: list[Any] = []
        self.values: dict[tuple[int, str, str], list[torch.Tensor]] = defaultdict(list)
        self.token_counts: dict[tuple[int, str, str], int] = defaultdict(int)

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

        def hook(_module: torch.nn.Module, inputs: Any) -> None:
            if inputs:
                self._store(layer, node, inputs[0])

        self.handles.append(module.register_forward_pre_hook(hook))

    def _store(self, layer: int, node: str, tensor: Any) -> None:
        context = self.context
        if context is None or not torch.is_tensor(tensor):
            return
        if tensor.dim() >= 3:
            tensor = tensor[0]
        if tensor.dim() != 2:
            return
        seq_len = min(int(tensor.shape[0]), len(context.modalities))
        tensor = tensor[:seq_len].detach().float().cpu()
        for modality in ("text", "sid_code"):
            positions = [idx for idx, item in enumerate(context.modalities[:seq_len]) if item == modality]
            if not positions:
                continue
            selected = tensor[positions].reshape(-1).contiguous()
            if self.max_values_per_modality > 0 and selected.numel() > self.max_values_per_modality:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    self.seed
                    + layer * 1009
                    + len(node) * 917
                    + len(modality) * 101
                    + self.token_counts[(layer, node, modality)]
                )
                indices = torch.randint(
                    selected.numel(),
                    (self.max_values_per_modality,),
                    generator=generator,
                    device="cpu",
                )
                selected = selected[indices].contiguous()
            self.values[(layer, node, modality)].append(selected)
            self.token_counts[(layer, node, modality)] += len(positions)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def register_hooks(model: torch.nn.Module, capture: ModalityActivationCapture) -> None:
    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        capture.add_output_hook(layer.input_layernorm, layer_idx, "attn_qkv_input")
        capture.add_input_hook(layer.self_attn.o_proj, layer_idx, "attn_o_input")
        capture.add_output_hook(layer.post_attention_layernorm, layer_idx, "ffn_gate_up_input")
        capture.add_input_hook(layer.mlp.down_proj, layer_idx, "ffn_down_input")
        capture.add_output_hook(layer, layer_idx, "block_output")


def concat_values(chunks: Sequence[torch.Tensor]) -> torch.Tensor:
    if not chunks:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat([chunk.reshape(-1).float() for chunk in chunks])


def finite_float(value: float) -> float | None:
    return value if math.isfinite(value) else None


def tensor_stats(values: torch.Tensor) -> dict[str, Any]:
    if values.numel() == 0:
        return {
            "num_values": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "mean_abs": None,
            "absmax": None,
            "p99_abs": None,
            "p999_abs": None,
        }
    x = values.float()
    abs_x = x.abs()
    return {
        "num_values": int(x.numel()),
        "mean": finite_float(float(x.mean().item())),
        "std": finite_float(float(x.std(unbiased=False).item())),
        "min": finite_float(float(x.min().item())),
        "max": finite_float(float(x.max().item())),
        "mean_abs": finite_float(float(abs_x.mean().item())),
        "absmax": finite_float(float(abs_x.max().item())),
        "p50_abs": finite_float(float(torch.quantile(abs_x, 0.50).item())),
        "p90_abs": finite_float(float(torch.quantile(abs_x, 0.90).item())),
        "p99_abs": finite_float(float(torch.quantile(abs_x, 0.99).item())),
        "p999_abs": finite_float(float(torch.quantile(abs_x, 0.999).item())),
    }


def sample_for_plot(values: torch.Tensor, max_values: int, seed: int) -> torch.Tensor:
    if values.numel() <= max_values:
        return values.float()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randint(values.numel(), (max_values,), generator=generator, device="cpu")
    return values[indices].float()


def ratio_or_none(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def build_summary(
    capture: ModalityActivationCapture,
    *,
    layers: Sequence[int],
    nodes: Sequence[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"entries": [], "gap_by_layer_node": []}
    stats_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    for layer in layers:
        for node in nodes:
            for modality in ("text", "sid_code"):
                key = (layer, node, modality)
                values = concat_values(capture.values.get(key, []))
                stats = tensor_stats(values)
                stats["layer"] = layer
                stats["node"] = node
                stats["modality"] = modality
                stats["num_tokens"] = int(capture.token_counts.get(key, 0))
                stats_by_key[key] = stats
                summary["entries"].append(stats)
            text_stats = stats_by_key[(layer, node, "text")]
            sid_stats = stats_by_key[(layer, node, "sid_code")]
            summary["gap_by_layer_node"].append(
                {
                    "layer": layer,
                    "node": node,
                    "sid_over_text_mean_abs": ratio_or_none(sid_stats["mean_abs"], text_stats["mean_abs"]),
                    "sid_over_text_p99_abs": ratio_or_none(sid_stats["p99_abs"], text_stats["p99_abs"]),
                    "sid_over_text_p999_abs": ratio_or_none(sid_stats["p999_abs"], text_stats["p999_abs"]),
                    "sid_over_text_absmax": ratio_or_none(sid_stats["absmax"], text_stats["absmax"]),
                    "text_num_tokens": text_stats["num_tokens"],
                    "sid_num_tokens": sid_stats["num_tokens"],
                }
            )
    return summary


def plot_signed_histograms(
    output_dir: Path,
    capture: ModalityActivationCapture,
    *,
    layers: Sequence[int],
    nodes: Sequence[str],
    bins: int,
    max_values: int,
    x_clip_quantile: float,
    seed: int,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    colors = {"text": "#1f77b4", "sid_code": "#2ca02c"}

    for node in nodes:
        ncols = min(3, len(layers))
        nrows = math.ceil(len(layers) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.1 * ncols, 3.6 * nrows), squeeze=False)
        legend_handles: dict[str, Any] = {}
        for ax, layer in zip(axes.flat, layers):
            raw_values = {
                modality: concat_values(capture.values.get((layer, node, modality), []))
                for modality in ("text", "sid_code")
            }
            non_empty = [value for value in raw_values.values() if value.numel() > 0]
            if not non_empty:
                ax.set_axis_off()
                continue
            combined = torch.cat(non_empty).float()
            abs_limit = float(torch.quantile(combined.abs(), x_clip_quantile).item()) if combined.numel() else 1.0
            abs_limit = max(abs_limit, 1e-6)
            for modality, values in raw_values.items():
                if values.numel() == 0:
                    continue
                sampled = sample_for_plot(values, max_values=max_values, seed=seed + layer + len(node) + len(modality))
                line_values = sampled.numpy()
                ax.hist(
                    line_values,
                    bins=bins,
                    range=(-abs_limit, abs_limit),
                    density=True,
                    histtype="step",
                    linewidth=1.4,
                    color=colors[modality],
                    label=modality,
                )
                legend_handles[modality] = ax.lines[-1] if ax.lines else None
            ax.set_title(f"Layer {layer} | {node}", fontsize=9)
            ax.set_xlabel("activation value")
            ax.set_ylabel("density")
            ax.grid(True, linewidth=0.3, alpha=0.35)
        for ax in axes.flat[len(layers) :]:
            ax.set_axis_off()
        handles = [handle for handle in legend_handles.values() if handle is not None]
        labels = [label for label, handle in legend_handles.items() if handle is not None]
        if handles:
            fig.legend(handles, labels, loc="upper right")
        fig.suptitle(f"Text vs SID-code activation distribution | {node}", fontsize=13)
        fig.tight_layout(rect=(0, 0, 0.95, 0.96))
        path = plot_dir / f"signed_distribution__{node}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(str(path))
    return written


def plot_abs_tail_curves(
    output_dir: Path,
    capture: ModalityActivationCapture,
    *,
    layers: Sequence[int],
    nodes: Sequence[str],
    max_values: int,
    seed: int,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    colors = {"text": "#1f77b4", "sid_code": "#2ca02c"}

    for node in nodes:
        ncols = min(3, len(layers))
        nrows = math.ceil(len(layers) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.1 * ncols, 3.6 * nrows), squeeze=False)
        legend_handles: dict[str, Any] = {}
        for ax, layer in zip(axes.flat, layers):
            for modality in ("text", "sid_code"):
                values = concat_values(capture.values.get((layer, node, modality), []))
                if values.numel() == 0:
                    continue
                sampled = sample_for_plot(values.abs(), max_values=max_values, seed=seed + 17 + layer + len(node) + len(modality))
                sampled = torch.sort(sampled.float()).values
                y = torch.linspace(1.0, 0.0, sampled.numel())
                line, = ax.plot(sampled.numpy(), y.numpy(), color=colors[modality], linewidth=1.3, label=modality)
                legend_handles[modality] = line
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1e-5, 1.0)
            ax.set_title(f"Layer {layer} | {node}", fontsize=9)
            ax.set_xlabel("|activation|")
            ax.set_ylabel("survival probability")
            ax.grid(True, which="both", linewidth=0.3, alpha=0.35)
        for ax in axes.flat[len(layers) :]:
            ax.set_axis_off()
        if legend_handles:
            fig.legend(list(legend_handles.values()), list(legend_handles.keys()), loc="upper right")
        fig.suptitle(f"Text vs SID-code activation tails | {node}", fontsize=13)
        fig.tight_layout(rect=(0, 0, 0.95, 0.96))
        path = plot_dir / f"abs_tail__{node}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(str(path))
    return written


def write_token_metadata(output_dir: Path, contexts: Sequence[TokenContext], prompt_stats: Mapping[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for context in contexts:
        for idx, (token_id, token_text, modality) in enumerate(
            zip(context.token_ids, context.token_texts, context.modalities)
        ):
            rows.append(
                {
                    "sample_id": context.sample_id,
                    "token_index": idx,
                    "token_id": token_id,
                    "token_text": token_text,
                    "modality": modality,
                }
            )
    (output_dir / "token_metadata.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "prompt_stats.json").write_text(json.dumps(prompt_stats, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    layers = parse_int_list(args.layers)
    nodes = parse_str_list(args.nodes)
    known_nodes = {"attn_qkv_input", "attn_o_input", "ffn_gate_up_input", "ffn_down_input", "block_output"}
    unknown = sorted(set(nodes) - known_nodes)
    if unknown:
        raise ValueError(f"Unknown nodes: {unknown}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    samples = load_ad_samples(
        tokenizer,
        str(resolve_repo_path(args.data_dir)),
        args.split,
        sample_size=args.sample_size,
        sample_offset=args.sample_offset,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype(args.dtype),
        device_map=None,
    ).to(device)
    model.eval()

    capture = ModalityActivationCapture(
        layers=set(layers),
        nodes=set(nodes),
        max_values_per_modality=args.capture_max_values_per_modality,
        seed=args.seed,
    )
    register_hooks(model, capture)

    contexts: list[TokenContext] = []
    prompt_stats: dict[str, Any] = {
        "sample_size": args.sample_size,
        "sample_offset": args.sample_offset,
        "max_history_sid_items": args.max_history_sid_items,
        "history_sid_keep": args.history_sid_keep,
        "samples": [],
    }
    try:
        for sample_id, sample in tqdm(samples, desc="capture prefill activations"):
            prompt, compression = compress_history_sid_items(
                sample["prompt"],
                max_items=args.max_history_sid_items,
                keep_policy=args.history_sid_keep,
            )
            prompt = effective_generation_prompt(prompt)
            input_ids = encode_prompt(tokenizer, prompt, args.max_tokens)
            token_ids = input_ids[0].tolist()
            token_texts = [safe_token_text(tokenizer, token_id) for token_id in token_ids]
            modalities = [token_modality(text, sid_boundary_as_sid=args.sid_boundary_as_sid) for text in token_texts]
            context = TokenContext(
                sample_id=sample_id,
                token_ids=token_ids,
                token_texts=token_texts,
                modalities=modalities,
            )
            contexts.append(context)
            prompt_stats["samples"].append(
                {
                    "sample_id": sample_id,
                    "prompt_tokens": len(token_ids),
                    "text_tokens": sum(1 for item in modalities if item == "text"),
                    "sid_code_tokens": sum(1 for item in modalities if item == "sid_code"),
                    "ignored_tokens": sum(1 for item in modalities if item is None),
                    "compression": compression,
                }
            )
            capture.context = context
            with torch.inference_mode():
                try:
                    model(input_ids=input_ids.to(device), use_cache=False)
                except TypeError:
                    model(input_ids=input_ids.to(device))
    finally:
        capture.context = None
        capture.close()

    summary = build_summary(capture, layers=layers, nodes=nodes)
    summary.update(
        {
            "paper_reference": "MQuant Figure 1(b) compares activation distributions of visual tokens and textual tokens in MLLMs.",
            "analogy": "OneRec history SID code tokens are treated as recommendation-modality tokens; ordinary prompt tokens are treated as text tokens.",
            "model_path": args.model_path,
            "data_dir": args.data_dir,
            "split": args.split,
            "sample_size": args.sample_size,
            "sample_offset": args.sample_offset,
            "layers": layers,
            "nodes": nodes,
            "dtype": args.dtype,
            "device": args.device,
            "max_history_sid_items": args.max_history_sid_items,
            "history_sid_keep": args.history_sid_keep,
            "max_tokens": args.max_tokens,
            "sid_boundary_as_sid": args.sid_boundary_as_sid,
            "capture_max_values_per_modality": args.capture_max_values_per_modality,
            "stats_note": "Activation value quantiles are computed from sampled values when a group exceeds capture_max_values_per_modality per sample.",
        }
    )
    write_token_metadata(output_dir, contexts, prompt_stats)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    signed_plots = plot_signed_histograms(
        output_dir,
        capture,
        layers=layers,
        nodes=nodes,
        bins=args.hist_bins,
        max_values=args.plot_max_values,
        x_clip_quantile=args.x_clip_quantile,
        seed=args.seed,
    )
    tail_plots: list[str] = []
    if not args.skip_abs_tail:
        tail_plots = plot_abs_tail_curves(
            output_dir,
            capture,
            layers=layers,
            nodes=nodes,
            max_values=args.plot_max_values,
            seed=args.seed,
        )
    summary["plots"] = {"signed_distribution": signed_plots, "abs_tail": tail_plots}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote modality activation distribution results to: {output_dir}")
    for item in summary["gap_by_layer_node"]:
        print(
            f"layer={item['layer']} node={item['node']} "
            f"sid/text p99_abs={item['sid_over_text_p99_abs']} "
            f"p999_abs={item['sid_over_text_p999_abs']} absmax={item['sid_over_text_absmax']}"
        )


if __name__ == "__main__":
    main()
