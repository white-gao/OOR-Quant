from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch

from real_quant.naive_w8a8.modules import FP8_MAX, activation_scale_per_token, quantize_fp8


def _import_vllm_ops():
    try:
        from vllm import _custom_ops as ops
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"Failed to import vLLM custom ops: {exc}") from exc
    if not hasattr(ops, "scaled_fp8_quant"):
        raise RuntimeError("vLLM custom ops does not expose scaled_fp8_quant.")
    return ops


def torch_dynamic_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = activation_scale_per_token(x, qmax=FP8_MAX, eps=1e-12)
    return quantize_fp8(x, scale, qmax=FP8_MAX), scale


def vllm_dynamic_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ops = _import_vllm_ops()
    y, scale = ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)
    if scale.ndim == 1:
        scale = scale.reshape(-1, 1)
    return y, scale


def bench(fn: Callable[[], tuple[torch.Tensor, torch.Tensor]], *, device: torch.device, warmup: int, reps: int) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        y = scale = None
        for _ in range(warmup):
            y, scale = fn()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        for _ in range(reps):
            y, scale = fn()
        end.record()
        torch.cuda.synchronize(device)
        wall_ms = (time.perf_counter() - wall_start) * 1000.0 / float(reps)
        cuda_ms = start.elapsed_time(end) / float(reps)
        assert y is not None and scale is not None
        return cuda_ms, wall_ms, y, scale


def max_abs_qdq_diff(x: torch.Tensor, y_a: torch.Tensor, scale_a: torch.Tensor, y_b: torch.Tensor, scale_b: torch.Tensor) -> tuple[float, float, float]:
    qdq_a = y_a.float() * scale_a.float()
    qdq_b = y_b.float() * scale_b.float()
    return (
        float((qdq_a - qdq_b).abs().max().item()),
        float((x.float() - qdq_a).abs().max().item()),
        float((x.float() - qdq_b).abs().max().item()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark torch dynamic FP8 activation quant vs vLLM fused scaled_fp8_quant.")
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument(
        "--shapes",
        nargs="*",
        default=("32x2048", "23744x2048", "32x6144", "23744x6144"),
        help="Activation shapes as MxK.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FP8 quant benchmark.")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(f"device={torch.cuda.get_device_name(device)} dtype={dtype} warmup={args.warmup} reps={args.reps}")
    _import_vllm_ops()
    print("shape,torch_cuda_ms,torch_wall_ms,vllm_cuda_ms,vllm_wall_ms,speedup_cuda,speedup_wall,torch_scale_shape,vllm_scale_shape,torch_y_dtype,vllm_y_dtype,qdq_diff_max,torch_qerr_max,vllm_qerr_max")
    for shape in args.shapes:
        m_str, k_str = shape.lower().split("x", 1)
        m, k = int(m_str), int(k_str)
        x = torch.randn(m, k, device=device, dtype=dtype)
        reps = args.reps
        if m * k >= 20_000_000:
            reps = max(20, min(args.reps, 50))
        torch_cuda, torch_wall, y_torch, scale_torch = bench(lambda: torch_dynamic_quant(x), device=device, warmup=args.warmup, reps=reps)
        vllm_cuda, vllm_wall, y_vllm, scale_vllm = bench(lambda: vllm_dynamic_quant(x), device=device, warmup=args.warmup, reps=reps)
        diff, qerr_torch, qerr_vllm = max_abs_qdq_diff(x, y_torch, scale_torch, y_vllm, scale_vllm)
        print(
            f"{m}x{k},{torch_cuda:.6f},{torch_wall:.6f},{vllm_cuda:.6f},{vllm_wall:.6f},"
            f"{torch_cuda / vllm_cuda if vllm_cuda > 0 else float('nan'):.4f},"
            f"{torch_wall / vllm_wall if vllm_wall > 0 else float('nan'):.4f},"
            f"{tuple(scale_torch.shape)},{tuple(scale_vllm.shape)},{y_torch.dtype},{y_vllm.dtype},"
            f"{diff:.6g},{qerr_torch:.6g},{qerr_vllm:.6g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
