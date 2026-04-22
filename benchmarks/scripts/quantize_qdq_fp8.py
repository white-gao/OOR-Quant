#!/usr/bin/env python3
"""
Create a normal HuggingFace checkpoint whose selected weights have gone through
FP8 e4m3 quantize-dequantize simulation.

This is an accuracy-risk experiment, not a real low-precision inference export:
the output tensors are saved back as regular floating point tensors so vLLM can
load the checkpoint through the existing BF16/FP16 path.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file


FP8_E4M3_MAX = 448.0
DEFAULT_TARGET_REGEX = r"model\.layers\.\d+\.mlp\.(gate_proj|up_proj)\.weight$"
SKIP_COPY_SUFFIXES = {".safetensors"}
SKIP_COPY_NAMES = {"model.safetensors.index.json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply FP8 e4m3 QDQ to selected safetensors weights."
    )
    parser.add_argument("--model_path", required=True, help="Input HF checkpoint directory.")
    parser.add_argument("--output_path", required=True, help="Output HF checkpoint directory.")
    parser.add_argument(
        "--target_regex",
        default=DEFAULT_TARGET_REGEX,
        help=f"Regex used to select tensors. Default: {DEFAULT_TARGET_REGEX}",
    )
    parser.add_argument(
        "--fp8_format",
        default="e4m3",
        choices=["e4m3"],
        help="FP8 format to simulate. Currently only e4m3 is supported.",
    )
    parser.add_argument(
        "--scale_granularity",
        default="per_row",
        choices=["none", "per_tensor", "per_row"],
        help=(
            "Scaling granularity before casting to FP8. "
            "'per_row' is usually a good first choice for Linear weights."
        ),
    )
    parser.add_argument(
        "--output_dtype",
        default="same",
        choices=["same", "bfloat16", "float16", "float32"],
        help="Dtype used to save QDQ tensors. Non-target tensors keep their original dtype.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used for QDQ math for matched tensors, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output_path if it already exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only list matched tensors and parameter counts; do not write output.",
    )
    return parser.parse_args()


def require_fp8_dtype() -> torch.dtype:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(
            "This PyTorch build does not expose torch.float8_e4m3fn. "
            "Please use a PyTorch version with native FP8 dtype support."
        )
    return torch.float8_e4m3fn


def resolve_output_dtype(output_dtype: str, source_dtype: torch.dtype) -> torch.dtype:
    if output_dtype == "same":
        return source_dtype
    if output_dtype == "bfloat16":
        return torch.bfloat16
    if output_dtype == "float16":
        return torch.float16
    if output_dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported output_dtype: {output_dtype}")


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def finite_or_none(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None


def make_scale(x: torch.Tensor, granularity: str) -> Tuple[torch.Tensor, Dict[str, Optional[float]]]:
    if granularity == "none":
        scale = torch.ones((), dtype=torch.float32, device=x.device)
    elif granularity == "per_tensor" or x.ndim < 2:
        max_abs = x.abs().amax()
        scale = torch.clamp(max_abs / FP8_E4M3_MAX, min=torch.finfo(torch.float32).tiny)
    elif granularity == "per_row":
        max_abs = x.abs().amax(dim=1, keepdim=True)
        scale = torch.clamp(max_abs / FP8_E4M3_MAX, min=torch.finfo(torch.float32).tiny)
    else:
        raise ValueError(f"Unsupported scale granularity: {granularity}")

    scale_float = scale.float()
    return scale, {
        "scale_min": finite_or_none(scalar(scale_float.min())),
        "scale_max": finite_or_none(scalar(scale_float.max())),
        "scale_mean": finite_or_none(scalar(scale_float.mean())),
    }


def fp8_e4m3_qdq(
    tensor: torch.Tensor,
    *,
    device: str,
    scale_granularity: str,
    output_dtype: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    fp8_dtype = require_fp8_dtype()
    source_dtype = tensor.dtype

    x = tensor.to(device=device, dtype=torch.float32, non_blocking=False)
    scale, scale_stats = make_scale(x, scale_granularity)

    x_scaled = x / scale
    x_scaled = torch.clamp(x_scaled, min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)
    x_qdq = x_scaled.to(fp8_dtype).to(torch.float32) * scale

    diff = x_qdq - x
    x_norm = torch.linalg.vector_norm(x)
    diff_norm = torch.linalg.vector_norm(diff)
    rel_l2 = diff_norm / torch.clamp(x_norm, min=torch.finfo(torch.float32).tiny)

    stats: Dict[str, Any] = {
        "source_dtype": str(source_dtype).replace("torch.", ""),
        "saved_dtype": str(resolve_output_dtype(output_dtype, source_dtype)).replace("torch.", ""),
        "scale_granularity": scale_granularity,
        "mse": finite_or_none(scalar((diff * diff).mean())),
        "mae": finite_or_none(scalar(diff.abs().mean())),
        "max_abs_error": finite_or_none(scalar(diff.abs().amax())),
        "relative_l2_error": finite_or_none(scalar(rel_l2)),
        "source_abs_max": finite_or_none(scalar(x.abs().amax())),
        "qdq_abs_max": finite_or_none(scalar(x_qdq.abs().amax())),
    }
    stats.update(scale_stats)

    save_dtype = resolve_output_dtype(output_dtype, source_dtype)
    return x_qdq.to(device="cpu", dtype=save_dtype), stats


def safetensor_files(model_path: Path) -> List[Path]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        names = sorted(set(index.get("weight_map", {}).values()))
        return [model_path / name for name in names]

    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found under {model_path}")
    return files


def iter_safetensor_keys(files: Iterable[Path]) -> Iterable[Tuple[Path, str, Tuple[int, ...], str, int]]:
    for file_path in files:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor_slice = f.get_slice(key)
                shape = tuple(tensor_slice.get_shape())
                numel = math.prod(shape)
                yield file_path, key, shape, tensor_slice.get_dtype(), numel


def copy_non_weight_files(model_path: Path, output_path: Path) -> None:
    for src in model_path.iterdir():
        if src.name in SKIP_COPY_NAMES or src.suffix in SKIP_COPY_SUFFIXES:
            continue
        dst = output_path / src.name
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst, follow_symlinks=True)


def prepare_output_dir(output_path: Path, overwrite: bool) -> None:
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {output_path}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=False)


def update_or_create_index(
    model_path: Path,
    output_path: Path,
    tensor_to_file: Dict[str, str],
    total_size: int,
) -> None:
    index_path = model_path / "model.safetensors.index.json"
    out_index_path = output_path / "model.safetensors.index.json"

    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
    else:
        index = {"metadata": {}, "weight_map": {}}

    index.setdefault("metadata", {})
    index["metadata"]["total_size"] = total_size
    index["weight_map"] = tensor_to_file

    with out_index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")


def build_report(
    *,
    args: argparse.Namespace,
    model_path: Path,
    output_path: Path,
    matched_reports: List[Dict[str, Any]],
    total_params: int,
    quantized_params: int,
    total_tensors: int,
    quantized_tensors: int,
    dry_run: bool,
) -> Dict[str, Any]:
    return {
        "model_path": str(model_path),
        "output_path": str(output_path),
        "target_regex": args.target_regex,
        "fp8_format": args.fp8_format,
        "fp8_max_finite": FP8_E4M3_MAX,
        "scale_granularity": args.scale_granularity,
        "output_dtype": args.output_dtype,
        "device": args.device,
        "dry_run": dry_run,
        "total_tensors": total_tensors,
        "quantized_tensors": quantized_tensors,
        "total_params": total_params,
        "quantized_params": quantized_params,
        "quantized_param_ratio": quantized_params / total_params if total_params else 0.0,
        "matched_tensors": matched_reports,
    }


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"model_path does not exist: {model_path}")

    pattern = re.compile(args.target_regex)
    files = safetensor_files(model_path)

    total_params = 0
    quantized_params = 0
    total_tensors = 0
    matched_reports: List[Dict[str, Any]] = []

    for file_path, key, shape, dtype, numel in iter_safetensor_keys(files):
        total_tensors += 1
        total_params += numel
        if pattern.search(key):
            quantized_params += numel
            matched_reports.append(
                {
                    "name": key,
                    "file": file_path.name,
                    "shape": list(shape),
                    "dtype": str(dtype).replace("torch.", ""),
                    "numel": numel,
                }
            )

    if not matched_reports:
        raise RuntimeError(
            f"No tensors matched --target_regex {args.target_regex!r}. "
            "Run with a broader regex or inspect model.named_parameters()/safetensors keys."
        )

    if args.dry_run:
        report = build_report(
            args=args,
            model_path=model_path,
            output_path=output_path,
            matched_reports=matched_reports,
            total_params=total_params,
            quantized_params=quantized_params,
            total_tensors=total_tensors,
            quantized_tensors=len(matched_reports),
            dry_run=True,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    prepare_output_dir(output_path, args.overwrite)
    copy_non_weight_files(model_path, output_path)

    tensor_to_file: Dict[str, str] = {}
    output_total_size = 0
    detailed_reports: List[Dict[str, Any]] = []

    with torch.no_grad():
        for file_path in files:
            output_tensors: Dict[str, torch.Tensor] = {}
            metadata: Optional[Dict[str, str]]

            with safe_open(file_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    if pattern.search(key):
                        qdq_tensor, stats = fp8_e4m3_qdq(
                            tensor,
                            device=args.device,
                            scale_granularity=args.scale_granularity,
                            output_dtype=args.output_dtype,
                        )
                        report_row = {
                            "name": key,
                            "file": file_path.name,
                            "shape": list(tensor.shape),
                            "numel": tensor.numel(),
                        }
                        report_row.update(stats)
                        detailed_reports.append(report_row)
                        output_tensors[key] = qdq_tensor
                        print(
                            f"[QDQ] {key} shape={tuple(tensor.shape)} "
                            f"mse={stats['mse']:.6e} rel_l2={stats['relative_l2_error']:.6e}"
                        )
                    else:
                        output_tensors[key] = tensor

                    tensor_to_file[key] = file_path.name
                    output_total_size += tensor_nbytes(output_tensors[key])

            save_file(output_tensors, output_path / file_path.name, metadata=metadata)
            del output_tensors

    update_or_create_index(model_path, output_path, tensor_to_file, output_total_size)

    report = build_report(
        args=args,
        model_path=model_path,
        output_path=output_path,
        matched_reports=detailed_reports,
        total_params=total_params,
        quantized_params=quantized_params,
        total_tensors=total_tensors,
        quantized_tensors=len(detailed_reports),
        dry_run=False,
    )

    report_path = output_path / "quant_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(
        f"Done. Quantized {quantized_params:,}/{total_params:,} parameters "
        f"({report['quantized_param_ratio']:.2%}) across {len(detailed_reports)} tensors."
    )
    print(f"Output checkpoint: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
