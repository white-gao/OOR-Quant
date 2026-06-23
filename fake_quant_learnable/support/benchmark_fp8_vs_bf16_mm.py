#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Callable

import torch


FP8_MAX = 448.0


@dataclass(frozen=True)
class ShapeSpec:
    name: str
    m: int
    k: int
    n: int


@dataclass(frozen=True)
class TimingStats:
    mean_ms: float
    p50_ms: float
    p90_ms: float


@dataclass(frozen=True)
class BenchResult:
    shape: ShapeSpec
    bf16: TimingStats
    fp8_gemm_only: TimingStats
    fp8_dynamic_act: TimingStats


PRESETS: dict[str, list[ShapeSpec]] = {
    "tiny": [
        ShapeSpec("decode_tiny", 32, 1024, 1024),
        ShapeSpec("prefill_tiny", 256, 1024, 1024),
    ],
    "transformer": [
        ShapeSpec("decode_hidden_32", 32, 4096, 4096),
        ShapeSpec("prefill_hidden_256", 256, 4096, 4096),
        ShapeSpec("prefill_hidden_1024", 1024, 4096, 4096),
        ShapeSpec("mlp_up_256", 256, 4096, 11008),
        ShapeSpec("mlp_down_256", 256, 11008, 4096),
    ],
    "large": [
        ShapeSpec("decode_hidden_32", 32, 4096, 4096),
        ShapeSpec("prefill_hidden_512", 512, 4096, 4096),
        ShapeSpec("prefill_hidden_2048", 2048, 4096, 4096),
        ShapeSpec("mlp_up_512", 512, 4096, 11008),
        ShapeSpec("mlp_down_512", 512, 11008, 4096),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Microbenchmark BF16 torch.mm vs FP8 torch._scaled_mm on CUDA. "
            "FP8 weights are treated as offline/pre-quantized; fp8_dynamic_act "
            "includes activation scalar absmax quantization per iteration."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="transformer")
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        metavar="NAME:M:K:N",
        help="Custom shape. Can be repeated, e.g. --shape decode:32:4096:4096.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    parser.add_argument(
        "--fast-accum",
        action="store_true",
        help="Pass use_fast_accum=True to torch._scaled_mm.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print available preset shapes and exit.",
    )
    return parser.parse_args()


def require_cuda_fp8(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError("This benchmark requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is required.")
    if not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("torch._scaled_mm is required.")


def parse_shape(value: str) -> ShapeSpec:
    parts = value.split(":")
    if len(parts) != 4:
        raise ValueError(f"Expected NAME:M:K:N, got {value!r}")
    name, m, k, n = parts
    return ShapeSpec(name=name, m=int(m), k=int(k), n=int(n))


def selected_shapes(args: argparse.Namespace) -> list[ShapeSpec]:
    shapes = list(PRESETS[args.preset])
    shapes.extend(parse_shape(item) for item in args.shape)
    return shapes


def scalar_absmax_scale(x: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    return (x.float().abs().amax().clamp_min(eps) / FP8_MAX).reshape(())


def quantize_fp8_scalar(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float() / scale, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def timed_cuda(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> TimingStats:
    if warmup < 0 or iters <= 0:
        raise ValueError(f"Invalid warmup/iters: {warmup}/{iters}")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        # Keep the result live until after the event is recorded.
        if out.numel() == 0:
            raise RuntimeError("Unexpected empty output.")
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    times_sorted = sorted(times)
    p90_idx = min(len(times_sorted) - 1, int(round(0.90 * (len(times_sorted) - 1))))
    return TimingStats(
        mean_ms=float(mean(times)),
        p50_ms=float(median(times)),
        p90_ms=float(times_sorted[p90_idx]),
    )


def benchmark_shape(
    shape: ShapeSpec,
    *,
    device: torch.device,
    warmup: int,
    iters: int,
    fast_accum: bool,
) -> BenchResult:
    a_bf16 = torch.randn(shape.m, shape.k, device=device, dtype=torch.bfloat16)
    # Store weight as [N, K], like nn.Linear.weight; GEMM uses weight.T as [K, N].
    weight_bf16 = torch.randn(shape.n, shape.k, device=device, dtype=torch.bfloat16)

    weight_scale = scalar_absmax_scale(weight_bf16)
    weight_fp8 = quantize_fp8_scalar(weight_bf16, weight_scale)
    weight_fp8_t = weight_fp8.t()

    act_scale = scalar_absmax_scale(a_bf16)
    a_fp8_static = quantize_fp8_scalar(a_bf16, act_scale)

    def bf16_mm() -> torch.Tensor:
        return torch.mm(a_bf16, weight_bf16.t())

    def fp8_gemm_only() -> torch.Tensor:
        return torch._scaled_mm(
            a_fp8_static,
            weight_fp8_t,
            scale_a=act_scale,
            scale_b=weight_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=fast_accum,
        )

    def fp8_dynamic_act() -> torch.Tensor:
        dynamic_scale = scalar_absmax_scale(a_bf16)
        a_fp8 = quantize_fp8_scalar(a_bf16, dynamic_scale)
        return torch._scaled_mm(
            a_fp8,
            weight_fp8_t,
            scale_a=dynamic_scale,
            scale_b=weight_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=fast_accum,
        )

    # One light correctness check so accidental shape/scale misuse is caught.
    bf16_out = bf16_mm()
    fp8_out = fp8_gemm_only()
    if bf16_out.shape != fp8_out.shape:
        raise RuntimeError(f"Output shape mismatch: {bf16_out.shape} vs {fp8_out.shape}")

    return BenchResult(
        shape=shape,
        bf16=timed_cuda(bf16_mm, warmup=warmup, iters=iters),
        fp8_gemm_only=timed_cuda(fp8_gemm_only, warmup=warmup, iters=iters),
        fp8_dynamic_act=timed_cuda(fp8_dynamic_act, warmup=warmup, iters=iters),
    )


def tflops(shape: ShapeSpec, mean_ms: float) -> float:
    flops = 2.0 * float(shape.m) * float(shape.k) * float(shape.n)
    return flops / (mean_ms / 1000.0) / 1e12


def print_preset_table() -> None:
    for preset, shapes in PRESETS.items():
        print(f"[{preset}]")
        for shape in shapes:
            print(f"  {shape.name}: M={shape.m}, K={shape.k}, N={shape.n}")


def print_results(results: list[BenchResult], *, device: torch.device) -> None:
    device_name = torch.cuda.get_device_name(device)
    print(f"device: {device} ({device_name})")
    print("note: FP8 uses torch._scaled_mm with scalar activation/weight scales.")
    print(
        "| shape | M | K | N | bf16 ms | fp8 gemm ms | fp8 dyn ms | "
        "gemm speedup | dyn speedup | bf16 TFLOP/s | fp8 TFLOP/s |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        s = result.shape
        bf16_ms = result.bf16.mean_ms
        fp8_ms = result.fp8_gemm_only.mean_ms
        dyn_ms = result.fp8_dynamic_act.mean_ms
        print(
            f"| {s.name} | {s.m} | {s.k} | {s.n} | "
            f"{bf16_ms:.4f} | {fp8_ms:.4f} | {dyn_ms:.4f} | "
            f"{bf16_ms / fp8_ms:.2f}x | {bf16_ms / dyn_ms:.2f}x | "
            f"{tflops(s, bf16_ms):.2f} | {tflops(s, fp8_ms):.2f} |"
        )


def write_csv(path: str | Path, results: list[BenchResult], *, device: torch.device) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "device",
                "shape",
                "m",
                "k",
                "n",
                "bf16_mean_ms",
                "bf16_p50_ms",
                "bf16_p90_ms",
                "fp8_gemm_mean_ms",
                "fp8_gemm_p50_ms",
                "fp8_gemm_p90_ms",
                "fp8_dynamic_mean_ms",
                "fp8_dynamic_p50_ms",
                "fp8_dynamic_p90_ms",
                "fp8_gemm_speedup",
                "fp8_dynamic_speedup",
                "bf16_tflops",
                "fp8_gemm_tflops",
            ],
        )
        writer.writeheader()
        for result in results:
            s = result.shape
            writer.writerow(
                {
                    "device": str(device),
                    "shape": s.name,
                    "m": s.m,
                    "k": s.k,
                    "n": s.n,
                    "bf16_mean_ms": result.bf16.mean_ms,
                    "bf16_p50_ms": result.bf16.p50_ms,
                    "bf16_p90_ms": result.bf16.p90_ms,
                    "fp8_gemm_mean_ms": result.fp8_gemm_only.mean_ms,
                    "fp8_gemm_p50_ms": result.fp8_gemm_only.p50_ms,
                    "fp8_gemm_p90_ms": result.fp8_gemm_only.p90_ms,
                    "fp8_dynamic_mean_ms": result.fp8_dynamic_act.mean_ms,
                    "fp8_dynamic_p50_ms": result.fp8_dynamic_act.p50_ms,
                    "fp8_dynamic_p90_ms": result.fp8_dynamic_act.p90_ms,
                    "fp8_gemm_speedup": result.bf16.mean_ms / result.fp8_gemm_only.mean_ms,
                    "fp8_dynamic_speedup": result.bf16.mean_ms / result.fp8_dynamic_act.mean_ms,
                    "bf16_tflops": tflops(s, result.bf16.mean_ms),
                    "fp8_gemm_tflops": tflops(s, result.fp8_gemm_only.mean_ms),
                }
            )


def main() -> None:
    args = parse_args()
    if args.list_presets:
        print_preset_table()
        return

    device = torch.device(args.device)
    require_cuda_fp8(device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    shapes = selected_shapes(args)
    results = [
        benchmark_shape(
            shape,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            fast_accum=args.fast_accum,
        )
        for shape in shapes
    ]
    print_results(results, device=device)
    if args.csv is not None:
        write_csv(args.csv, results, device=device)
        print(f"wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
