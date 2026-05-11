#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open


DEFAULT_MODEL_PATH = "/home/guowei/OneRec-1.7B"
DEFAULT_OUTPUT_DIR = "fake_quant/weight_probe/results/v1.0/OneRec-1.7B"
@dataclass
class TensorStats:
    name: str
    shape: List[int]
    dtype: str
    numel: int
    min: float
    max: float
    mean: float
    std: float
    absmax: float
    p001: float
    p01: float
    p05: float
    p50: float
    p95: float
    p99: float
    p999: float
    abs_p99: float
    abs_p999: float
    zero_ratio: float
    sample_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot OneRec weight distributions.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample_per_tensor", type=int, default=2_000_000)
    parser.add_argument("--chunk_elements", type=int, default=8_000_000)
    parser.add_argument("--hist_bins", type=int, default=400)
    parser.add_argument("--include_embedding", action="store_true")
    parser.add_argument(
        "--targets",
        default="default",
        help='Target preset: "default", "all", or comma-separated regex patterns. '
        'Default selects all tensors except embedding unless --include_embedding is set.',
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_weight_map(model_path: Path) -> Dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]

    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files found under {model_path}")
    weight_map: Dict[str, str] = {}
    for file_path in files:
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                weight_map[key] = file_path.name
    return weight_map


def select_targets(
    weight_names: Iterable[str],
    target_spec: str,
    *,
    include_embedding: bool,
) -> List[str]:
    names = sorted(weight_names, key=natural_sort_key)
    if target_spec == "all":
        selected = names
    elif target_spec != "default":
        patterns = [re.compile(item.strip()) for item in target_spec.split(",") if item.strip()]
        selected = [name for name in names if any(pattern.search(name) for pattern in patterns)]
    else:
        selected = names

    if not include_embedding:
        selected = [name for name in selected if name != "model.embed_tokens.weight"]
    return sorted(set(selected), key=natural_sort_key)


def natural_sort_key(text: str) -> List[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def safe_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


def tensor_output_dir(base_dir: Path, tensor_name: str) -> Path:
    layer_match = re.search(r"model\.layers\.(\d+)\.", tensor_name)
    if tensor_name == "model.embed_tokens.weight":
        return base_dir / "embedding"
    if layer_match:
        return base_dir / f"layer_{int(layer_match.group(1)):02d}"
    return base_dir / "other"


def collect_sample_and_stats(
    tensor: torch.Tensor,
    *,
    sample_per_tensor: int,
    chunk_elements: int,
    seed: int,
) -> tuple[TensorStats, np.ndarray]:
    flat = tensor.reshape(-1)
    numel = int(flat.numel())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    if numel <= sample_per_tensor:
        sample_indices = torch.arange(numel)
    else:
        sample_indices = torch.randint(numel, (sample_per_tensor,), generator=generator)

    sample = flat[sample_indices].float().cpu().numpy()
    total_sum = 0.0
    total_sumsq = 0.0
    total_min = math.inf
    total_max = -math.inf
    zero_count = 0

    for start in range(0, numel, chunk_elements):
        end = min(start + chunk_elements, numel)
        chunk = flat[start:end].float()
        total_sum += float(chunk.sum().item())
        total_sumsq += float((chunk * chunk).sum().item())
        total_min = min(total_min, float(chunk.min().item()))
        total_max = max(total_max, float(chunk.max().item()))
        zero_count += int((chunk == 0).sum().item())
        del chunk

    mean = total_sum / numel
    variance = max(total_sumsq / numel - mean * mean, 0.0)
    std = math.sqrt(variance)
    quantiles = np.quantile(sample, [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
    abs_sample = np.abs(sample)

    stats = TensorStats(
        name="",
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        numel=numel,
        min=total_min,
        max=total_max,
        mean=mean,
        std=std,
        absmax=max(abs(total_min), abs(total_max)),
        p001=float(quantiles[0]),
        p01=float(quantiles[1]),
        p05=float(quantiles[2]),
        p50=float(quantiles[3]),
        p95=float(quantiles[4]),
        p99=float(quantiles[5]),
        p999=float(quantiles[6]),
        abs_p99=float(np.quantile(abs_sample, 0.99)),
        abs_p999=float(np.quantile(abs_sample, 0.999)),
        zero_ratio=zero_count / numel,
        sample_size=int(sample.shape[0]),
    )
    return stats, sample


def plot_distribution(
    sample: np.ndarray,
    stats: TensorStats,
    output_path: Path,
    *,
    hist_bins: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=160)

    lower, upper = np.quantile(sample, [0.0005, 0.9995])
    if lower == upper:
        lower = float(sample.min())
        upper = float(sample.max())
    if lower == upper:
        lower -= 1.0
        upper += 1.0

    counts, edges = np.histogram(sample, bins=hist_bins, range=(lower, upper), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    ax.plot(centers, counts, linewidth=1.4, color="#1f6f8b")
    ax.axvline(0.0, color="#555555", linewidth=0.8, alpha=0.6)
    ax.axvline(stats.mean, color="#c0392b", linewidth=0.9, linestyle="--", label="mean")
    ax.set_title(stats.name, fontsize=10)
    ax.set_xlabel("weight value")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    text = (
        f"shape={tuple(stats.shape)}\n"
        f"dtype={stats.dtype}, n={stats.numel:,}\n"
        f"mean={stats.mean:.4g}, std={stats.std:.4g}\n"
        f"min={stats.min:.4g}, max={stats.max:.4g}\n"
        f"p99={stats.p99:.4g}, abs_p999={stats.abs_p999:.4g}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82, "linewidth": 0.4},
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_stats(stats_rows: List[TensorStats], stats_dir: Path) -> None:
    stats_dir.mkdir(parents=True, exist_ok=True)
    rows = [row.__dict__ for row in stats_rows]
    (stats_dir / "weight_stats.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (stats_dir / "weight_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    stats_dir = output_dir / "stats"

    weight_map = load_weight_map(model_path)
    targets = select_targets(
        weight_map.keys(),
        args.targets,
        include_embedding=args.include_embedding,
    )
    if not targets:
        raise ValueError(f"No weight tensors selected by targets={args.targets!r}")

    tensors_by_file: Dict[str, List[str]] = {}
    for name in targets:
        tensors_by_file.setdefault(weight_map[name], []).append(name)

    stats_rows: List[TensorStats] = []
    for file_name, names in sorted(tensors_by_file.items()):
        file_path = model_path / file_name
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            for tensor_name in sorted(names, key=natural_sort_key):
                print(f"Processing {tensor_name}", flush=True)
                tensor = handle.get_tensor(tensor_name)
                stats, sample = collect_sample_and_stats(
                    tensor,
                    sample_per_tensor=args.sample_per_tensor,
                    chunk_elements=args.chunk_elements,
                    seed=args.seed + len(stats_rows),
                )
                stats.name = tensor_name
                stats_rows.append(stats)
                output_path = tensor_output_dir(plots_dir, tensor_name) / f"{safe_name(tensor_name)}.png"
                plot_distribution(sample, stats, output_path, hist_bins=args.hist_bins)
                del tensor

    write_stats(stats_rows, stats_dir)
    print(f"Saved {len(stats_rows)} distribution plots to: {plots_dir}")
    print(f"Saved stats to: {stats_dir}")


if __name__ == "__main__":
    main()
