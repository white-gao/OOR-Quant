#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
for path in (PROJECT_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.tasks.v1_0.registry import get_loader, get_task_config


DEFAULT_MODEL_PATH = "/home/guowei/OneRec-1.7B"
DEFAULT_DATA_DIR = "data/onerec_data/benchmark-data"
DEFAULT_OUTPUT_DIR = "fake_quant/activation_profiles/v1.0/OneRec-1.7B-ad-sample-32"
FALLBACK_MODEL_PATHS = [
    "/zssd/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B",
    "/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B",
]
FALLBACK_DATA_DIRS = [
    "/zssd/home/yhhuang/Projects/OOR-Quant/data/onerec_data/benchmark-data",
    "/zssd/home/yhhuang/Projects/OOR-Quant/data/onerec_data/benchmark-data",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile OneRec AD activation outliers with HF inference.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample_size", default="32")
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_decode_steps", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=0, help="Optional left truncation for long prompts.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--outlier_thresholds",
        default="6,10,20",
        help="Comma-separated absolute activation thresholds for outlier ratios.",
    )
    parser.add_argument(
        "--save_histograms",
        action="store_true",
        help="Save log-binned activation histograms and layer-wise distribution plots.",
    )
    parser.add_argument(
        "--hist_modules",
        default="residual_block_output,mlp.down_proj,self_attn.q_proj,self_attn.k_proj,self_attn.v_proj",
        help="Comma-separated modules to include in histogram plots. Use 'all' for all hooked modules.",
    )
    parser.add_argument(
        "--hist_stages",
        default="prefill,decode_step_1,decode_step_2,decode_step_3",
        help="Comma-separated stages to include in histogram plots. Use 'all' for all stages.",
    )
    parser.add_argument("--hist_bins", type=int, default=120)
    parser.add_argument("--hist_log2_min", type=float, default=-12.0)
    parser.add_argument("--hist_log2_max", type=float, default=14.0)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def parse_sample_size(value: str) -> Any:
    if value == "" or value.lower() == "none":
        return None
    if value == "full":
        return "full"
    return int(value)


def parse_thresholds(value: str) -> List[float]:
    thresholds = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not thresholds:
        raise ValueError("--outlier_thresholds must contain at least one value")
    return thresholds


def parse_csv_filter(value: str) -> Optional[set[str]]:
    items = {part.strip() for part in value.split(",") if part.strip()}
    if not items or "all" in items:
        return None
    return items


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

    raise FileNotFoundError(
        f"Model path not found: {model_path}. "
        "Set MODEL_PATH or pass --model_path to a local OneRec-1.7B HF checkpoint."
    )


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

    raise FileNotFoundError(
        f"AD benchmark data not found under: {data_dir}. "
        "Set DATA_DIR or pass --data_dir to a directory containing ad/ad_test.parquet."
    )


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


def finite_or_none(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None


def tensor_event_stats(values: torch.Tensor, thresholds: Sequence[float]) -> Dict[str, Any]:
    flat = values.detach().float().abs().flatten()
    if flat.numel() == 0:
        out = {
            "numel": 0,
            "mean_abs": None,
            "std_abs": None,
            "absmax": None,
            "p99": None,
            "p999": None,
            "absmax_over_p99": None,
            "absmax_over_p999": None,
        }
        for threshold in thresholds:
            out[f"outlier_ratio_gt_{threshold:g}"] = None
        return out

    mean_abs = float(flat.mean().item())
    std_abs = float(flat.std(unbiased=False).item())
    absmax = float(flat.max().item())
    p99 = float(torch.quantile(flat, 0.99).item())
    p999 = float(torch.quantile(flat, 0.999).item())
    eps = torch.finfo(torch.float32).tiny

    out = {
        "numel": int(flat.numel()),
        "mean_abs": finite_or_none(mean_abs),
        "std_abs": finite_or_none(std_abs),
        "absmax": finite_or_none(absmax),
        "p99": finite_or_none(p99),
        "p999": finite_or_none(p999),
        "absmax_over_p99": finite_or_none(absmax / max(p99, eps)),
        "absmax_over_p999": finite_or_none(absmax / max(p999, eps)),
    }
    for threshold in thresholds:
        out[f"outlier_ratio_gt_{threshold:g}"] = float((flat > threshold).float().mean().item())
    return out


@dataclass
class ForwardContext:
    sample_id: str
    stage: str
    token_groups: List[str]
    collect_all_groups: bool
    last_token_stage: Optional[str] = None
    last_token_group: Optional[str] = None


class ActivationProfiler:
    def __init__(
        self,
        thresholds: Sequence[float],
        histogram_store: Optional["ActivationHistogramStore"] = None,
    ) -> None:
        self.thresholds = list(thresholds)
        self.histogram_store = histogram_store
        self.rows: List[Dict[str, Any]] = []
        self.handles: List[Any] = []
        self.context: Optional[ForwardContext] = None

    def add_hook(self, module: torch.nn.Module, layer: int, module_name: str) -> None:
        def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            context = self.context
            if context is None:
                return

            tensor = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(tensor) or tensor.dim() < 2:
                return

            if tensor.dim() >= 3:
                tensor = tensor[0]
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)

            seq_len = min(int(tensor.shape[0]), len(context.token_groups))
            if seq_len <= 0:
                return

            if context.collect_all_groups:
                groups = sorted(set(context.token_groups[:seq_len]))
                for group in groups:
                    indices = [idx for idx, item in enumerate(context.token_groups[:seq_len]) if item == group]
                    if indices:
                        self._append_row(
                            tensor=tensor[indices],
                            layer=layer,
                            module_name=module_name,
                            context=context,
                            stage=context.stage,
                            group=group,
                            positions=len(indices),
                        )

            if context.last_token_stage is not None:
                self._append_row(
                    tensor=tensor[seq_len - 1],
                    layer=layer,
                    module_name=module_name,
                    context=context,
                    stage=context.last_token_stage,
                    group=context.last_token_group or context.token_groups[seq_len - 1],
                    positions=1,
                )

            if not context.collect_all_groups and context.last_token_stage is None:
                self._append_row(
                    tensor=tensor[:seq_len],
                    layer=layer,
                    module_name=module_name,
                    context=context,
                    stage=context.stage,
                    group=context.token_groups[seq_len - 1],
                    positions=seq_len,
                )

        self.handles.append(module.register_forward_hook(hook))

    def _append_row(
        self,
        *,
        tensor: torch.Tensor,
        layer: int,
        module_name: str,
        context: ForwardContext,
        stage: str,
        group: str,
        positions: int,
    ) -> None:
        row = {
            "sample_id": context.sample_id,
            "layer": layer,
            "module": module_name,
            "stage": stage,
            "token_group": group,
            "positions": positions,
        }
        row.update(tensor_event_stats(tensor, self.thresholds))
        self.rows.append(row)
        if self.histogram_store is not None:
            self.histogram_store.add(
                tensor=tensor,
                layer=layer,
                module=module_name,
                stage=stage,
                token_group=group,
            )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class ActivationHistogramStore:
    def __init__(
        self,
        *,
        modules: Optional[set[str]],
        stages: Optional[set[str]],
        bins: int,
        log2_min: float,
        log2_max: float,
    ) -> None:
        if bins <= 0:
            raise ValueError("--hist_bins must be positive")
        if log2_min >= log2_max:
            raise ValueError("--hist_log2_min must be smaller than --hist_log2_max")
        self.modules = modules
        self.stages = stages
        self.bins = bins
        self.log2_min = log2_min
        self.log2_max = log2_max
        self.counts: Dict[tuple, torch.Tensor] = {}
        self.total: Dict[tuple, int] = collections.Counter()

    def add(
        self,
        *,
        tensor: torch.Tensor,
        layer: int,
        module: str,
        stage: str,
        token_group: str,
    ) -> None:
        if self.modules is not None and module not in self.modules:
            return
        if self.stages is not None and stage not in self.stages:
            return

        flat = tensor.detach().float().abs().flatten()
        if flat.numel() == 0:
            return

        log_values = torch.log2(flat.clamp_min(2.0 ** self.log2_min))
        log_values = log_values.clamp(min=self.log2_min, max=self.log2_max)
        hist = torch.histc(
            log_values.cpu(),
            bins=self.bins,
            min=self.log2_min,
            max=self.log2_max,
        ).to(torch.float64)

        key = (int(layer), module, stage, token_group)
        if key not in self.counts:
            self.counts[key] = torch.zeros(self.bins, dtype=torch.float64)
        self.counts[key] += hist
        self.total[key] += int(flat.numel())

    def bin_edges(self) -> List[float]:
        step = (self.log2_max - self.log2_min) / self.bins
        return [self.log2_min + i * step for i in range(self.bins + 1)]

    def to_rows(self) -> List[Dict[str, Any]]:
        edges = self.bin_edges()
        rows: List[Dict[str, Any]] = []
        for key, counts in sorted(self.counts.items()):
            layer, module, stage, token_group = key
            total = max(self.total[key], 1)
            count_values = counts.tolist()
            for idx, count in enumerate(count_values):
                left_log2 = edges[idx]
                right_log2 = edges[idx + 1]
                center_log2 = (left_log2 + right_log2) / 2.0
                rows.append(
                    {
                        "layer": layer,
                        "module": module,
                        "stage": stage,
                        "token_group": token_group,
                        "bin_index": idx,
                        "left_abs": 2.0 ** left_log2,
                        "right_abs": 2.0 ** right_log2,
                        "center_abs": 2.0 ** center_log2,
                        "count": int(count),
                        "probability": float(count / total),
                        "total": total,
                    }
                )
        return rows

    def plot(self, output_dir: Path) -> List[str]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        edges = self.bin_edges()
        centers = [2.0 ** ((edges[idx] + edges[idx + 1]) / 2.0) for idx in range(self.bins)]

        groups_by_stage_module: Dict[tuple, Dict[int, Dict[str, torch.Tensor]]] = {}
        for key, counts in self.counts.items():
            layer, module, stage, token_group = key
            groups_by_stage_module.setdefault((stage, module), {}).setdefault(layer, {})[token_group] = counts

        written: List[str] = []
        colors = {
            "chat_special": "#d62728",
            "prompt_text": "#1f77b4",
            "sid_boundary": "#ff7f0e",
            "sid_code": "#2ca02c",
            "sid_boundary_next_token": "#9467bd",
            "generated_sid_code": "#17becf",
        }
        preferred_order = [
            "chat_special",
            "prompt_text",
            "sid_boundary",
            "sid_code",
            "sid_boundary_next_token",
            "generated_sid_code",
        ]

        for (stage, module), by_layer in sorted(groups_by_stage_module.items()):
            layers = sorted(by_layer)
            if not layers:
                continue

            ncols = 4
            nrows = math.ceil(len(layers) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.2 * nrows), squeeze=False)
            legend_handles = {}

            for ax, layer in zip(axes.flat, layers):
                group_counts = by_layer[layer]
                groups = [group for group in preferred_order if group in group_counts]
                groups.extend(sorted(group for group in group_counts if group not in groups))

                for group in groups:
                    counts = group_counts[group]
                    total = max(float(counts.sum().item()), 1.0)
                    probs = [value / total for value in counts.tolist()]
                    line, = ax.plot(
                        centers,
                        probs,
                        label=group,
                        color=colors.get(group),
                        linewidth=1.2,
                        alpha=0.9,
                    )
                    legend_handles[group] = line

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(f"Layer {layer}", fontsize=9)
                ax.grid(True, which="both", linewidth=0.3, alpha=0.35)
                ax.tick_params(axis="both", labelsize=7)

            for ax in axes.flat[len(layers) :]:
                ax.axis("off")

            fig.suptitle(f"{stage} / {module} | abs activation distribution", fontsize=14)
            fig.supxlabel("|activation|")
            fig.supylabel("probability per log2 bin")
            if legend_handles:
                fig.legend(
                    [legend_handles[group] for group in preferred_order if group in legend_handles]
                    + [handle for group, handle in legend_handles.items() if group not in preferred_order],
                    [group for group in preferred_order if group in legend_handles]
                    + [group for group in legend_handles if group not in preferred_order],
                    loc="upper right",
                    fontsize=9,
                )
            fig.tight_layout(rect=(0, 0, 0.95, 0.97))

            safe_name = f"{stage}__{module}".replace(".", "_").replace("/", "_")
            path = plot_dir / f"{safe_name}_by_layer.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            written.append(str(path))

        return written


def register_hooks(model: torch.nn.Module, profiler: ActivationProfiler) -> None:
    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        profiler.add_hook(layer.input_layernorm, layer_idx, "attn_input_norm")
        profiler.add_hook(layer.post_attention_layernorm, layer_idx, "mlp_input_norm")
        profiler.add_hook(layer, layer_idx, "residual_block_output")

        self_attn = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if hasattr(self_attn, name):
                profiler.add_hook(getattr(self_attn, name), layer_idx, f"self_attn.{name}")

        mlp = layer.mlp
        for name in ("gate_proj", "up_proj", "down_proj"):
            if hasattr(mlp, name):
                profiler.add_hook(getattr(mlp, name), layer_idx, f"mlp.{name}")

    profiler.add_hook(model.model.norm, len(layers), "final_norm")


def load_ad_data(tokenizer: Any, data_dir: str, split: str, sample_size: Any) -> Dict[str, Dict[str, Any]]:
    loader = get_loader(
        task_name="ad",
        data_dir=data_dir,
        tokenizer=tokenizer,
        enable_thinking=False,
    )
    return loader.load_data(split=split, sample_size=sample_size)


def encode_prompt(
    tokenizer: Any,
    prompt: str,
    prompt_token: str,
    max_tokens: int,
) -> torch.Tensor:
    if prompt_token and not prompt.endswith(prompt_token):
        prompt = prompt + prompt_token
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if max_tokens and input_ids.shape[1] > max_tokens:
        input_ids = input_ids[:, -max_tokens:]
    return input_ids


def make_prefill_groups(tokenizer: Any, token_ids: List[int]) -> List[str]:
    return [token_group(safe_token_text(tokenizer, token_id)) for token_id in token_ids]


def generated_group(tokenizer: Any, token_id: int) -> str:
    group = token_group(safe_token_text(tokenizer, token_id))
    return f"generated_{group}"


def profile_sample(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    profiler: ActivationProfiler,
    sample_id: str,
    prompt: str,
    prompt_token: str,
    device: str,
    max_tokens: int,
    num_decode_steps: int,
) -> Dict[str, Any]:
    input_ids = encode_prompt(tokenizer, prompt, prompt_token, max_tokens)
    token_ids = input_ids[0].tolist()
    token_groups = make_prefill_groups(tokenizer, token_ids)
    input_ids = input_ids.to(device)

    generated_ids: List[int] = []
    generated_texts: List[str] = []

    profiler.context = ForwardContext(
        sample_id=sample_id,
        stage="prefill",
        token_groups=token_groups,
        collect_all_groups=True,
        last_token_stage="decode_step_1" if num_decode_steps > 0 else None,
        last_token_group="sid_boundary_next_token",
    )

    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True)
        next_id = int(torch.argmax(output.logits[:, -1, :], dim=-1).item())
        past_key_values = output.past_key_values

        for step in range(1, num_decode_steps + 1):
            generated_ids.append(next_id)
            generated_texts.append(safe_token_text(tokenizer, next_id))

            if step == num_decode_steps:
                break

            step_input = torch.tensor([[next_id]], device=device)
            profiler.context = ForwardContext(
                sample_id=sample_id,
                stage=f"decode_step_{step + 1}",
                token_groups=[generated_group(tokenizer, next_id)],
                collect_all_groups=False,
            )
            output = model(input_ids=step_input, past_key_values=past_key_values, use_cache=True)
            next_id = int(torch.argmax(output.logits[:, -1, :], dim=-1).item())
            past_key_values = output.past_key_values

    profiler.context = None

    return {
        "sample_id": sample_id,
        "prompt_tokens": len(token_ids),
        "prefill_token_groups": {group: token_groups.count(group) for group in sorted(set(token_groups))},
        "generated_ids": generated_ids,
        "generated_texts": generated_texts,
    }


def mean(values: Sequence[float]) -> Optional[float]:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def aggregate_rows(rows: Iterable[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    buckets: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(tuple(row[key] for key in keys), []).append(row)

    metric_columns = [
        "mean_abs",
        "std_abs",
        "absmax",
        "p99",
        "p999",
        "absmax_over_p99",
        "absmax_over_p999",
    ]
    threshold_columns = [key for key in rows[0].keys() if key.startswith("outlier_ratio_gt_")] if rows else []

    out = []
    for bucket_key, bucket_rows in sorted(buckets.items()):
        item = {key: value for key, value in zip(keys, bucket_key)}
        item["events"] = len(bucket_rows)
        item["positions"] = sum(int(row.get("positions", 0)) for row in bucket_rows)
        item["numel"] = sum(int(row.get("numel", 0)) for row in bucket_rows)

        for column in metric_columns + threshold_columns:
            values = [row.get(column) for row in bucket_rows]
            item[f"{column}_avg"] = mean(values)
            clean = [value for value in values if value is not None and math.isfinite(value)]
            item[f"{column}_max"] = max(clean) if clean else None

        out.append(item)
    return out


def top_outlier_rows(rows: List[Dict[str, Any]], limit: int = 200) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            row.get("absmax_over_p999") or 0.0,
            row.get("absmax") or 0.0,
        ),
        reverse=True,
    )[:limit]


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
            writer.writerow(row)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    output_dir = Path(args.output_dir)
    thresholds = parse_thresholds(args.outlier_thresholds)
    histogram_store = None
    if args.save_histograms:
        histogram_store = ActivationHistogramStore(
            modules=parse_csv_filter(args.hist_modules),
            stages=parse_csv_filter(args.hist_stages),
            bins=args.hist_bins,
            log2_min=args.hist_log2_min,
            log2_max=args.hist_log2_max,
        )

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

    profiler = ActivationProfiler(thresholds, histogram_store=histogram_store)
    register_hooks(model, profiler)

    start = time.time()
    sample_summaries = []
    try:
        for sample_id, sample in tqdm(test_data.items(), desc="Profiling AD activations"):
            sample_summaries.append(
                profile_sample(
                    model=model,
                    tokenizer=tokenizer,
                    profiler=profiler,
                    sample_id=sample_id,
                    prompt=sample["prompt"],
                    prompt_token=prompt_token,
                    device=args.device,
                    max_tokens=args.max_tokens,
                    num_decode_steps=args.num_decode_steps,
                )
            )
    finally:
        profiler.close()

    elapsed = time.time() - start

    rows = profiler.rows
    write_csv(output_dir / "event_stats.csv", rows)
    write_csv(output_dir / "summary_by_layer_module.csv", aggregate_rows(rows, ["layer", "module"]))
    write_csv(output_dir / "summary_by_stage_module.csv", aggregate_rows(rows, ["stage", "module"]))
    write_csv(output_dir / "summary_by_stage_token_group_module.csv", aggregate_rows(rows, ["stage", "token_group", "module"]))
    write_csv(output_dir / "top_outliers.csv", top_outlier_rows(rows))
    histogram_plot_paths: List[str] = []
    if histogram_store is not None:
        write_csv(output_dir / "activation_histograms.csv", histogram_store.to_rows())
        histogram_plot_paths = histogram_store.plot(output_dir)
    write_json(
        output_dir / "sample_summary.json",
        {
            "config": vars(args),
            "resolved_model_path": model_path,
            "resolved_data_dir": data_dir,
            "num_samples": len(test_data),
            "num_event_rows": len(rows),
            "elapsed_seconds": elapsed,
            "outlier_thresholds": thresholds,
            "histogram_plots": histogram_plot_paths,
            "samples": sample_summaries,
        },
    )

    print(f"Samples: {len(test_data)}")
    print(f"Event rows: {len(rows)}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Wrote activation profile to: {output_dir}")
    if histogram_store is not None:
        print(f"Histogram plots: {len(histogram_plot_paths)}")
        print(f"Wrote histogram CSV to: {output_dir / 'activation_histograms.csv'}")
        print(f"Wrote plots to: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
