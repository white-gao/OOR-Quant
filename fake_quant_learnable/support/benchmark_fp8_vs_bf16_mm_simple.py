#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import mean, median
from typing import Callable

import torch

FP8_MAX = 448.0


@dataclass(frozen=True)
class Shape:
    name: str
    m: int
    k: int
    n: int


DEFAULT_SHAPES = [
    Shape('small_decode', 32, 1024, 1024),
    Shape('small_prefill', 256, 1024, 1024),
    Shape('decode_hidden', 32, 4096, 4096),
    Shape('prefill_256', 256, 4096, 4096),
    Shape('prefill_1024', 1024, 4096, 4096),
    Shape('mlp_up_256', 256, 4096, 11008),
    Shape('mlp_down_256', 256, 11008, 4096),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Simple BF16 vs FP8 matrix multiplication benchmark. Quantization is outside timing.'
    )
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--shape',
        action='append',
        default=[],
        metavar='NAME:M:K:N',
        help='Custom shape. Can be repeated, e.g. --shape test:256:4096:4096.',
    )
    return parser.parse_args()


def parse_shape(value: str) -> Shape:
    parts = value.split(':')
    if len(parts) != 4:
        raise ValueError(f'Expected NAME:M:K:N, got {value!r}')
    name, m, k, n = parts
    return Shape(name, int(m), int(k), int(n))


def fp8_scale(x: torch.Tensor) -> torch.Tensor:
    # Scalar scale keeps this benchmark simple and matches what torch._scaled_mm
    # accepts on the current PyTorch/CUDA stack.
    return (x.float().abs().amax().clamp_min(1e-12) / FP8_MAX).reshape(())


def to_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float() / scale, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def time_cuda(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        if out.numel() == 0:
            raise RuntimeError('empty output')
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return float(mean(times)), float(median(times))


def tflops(shape: Shape, ms: float) -> float:
    return 2.0 * shape.m * shape.k * shape.n / (ms / 1000.0) / 1e12


def run_one(shape: Shape, *, device: torch.device, warmup: int, iters: int) -> None:
    # Define two BF16 tensors for the reference matmul.
    a_bf16 = torch.randn(shape.m, shape.k, device=device, dtype=torch.bfloat16)
    b_bf16 = torch.randn(shape.k, shape.n, device=device, dtype=torch.bfloat16)

    # Define the corresponding FP8 tensors once, outside timing.
    # FP8 GEMM needs scales to interpret the stored FP8 values.
    a_scale = fp8_scale(a_bf16)
    b_scale = fp8_scale(b_bf16)
    a_fp8 = to_fp8(a_bf16, a_scale)
    # cuBLASLt FP8 scaled_mm requires a row-major lhs and column-major rhs.
    # Build rhs from [N, K] contiguous storage, then view it as [K, N].
    b_fp8 = to_fp8(b_bf16.t().contiguous(), b_scale).t()

    def bf16_mm() -> torch.Tensor:
        return torch.mm(a_bf16, b_bf16)

    def fp8_mm() -> torch.Tensor:
        return torch._scaled_mm(
            a_fp8,
            b_fp8,
            scale_a=a_scale,
            scale_b=b_scale,
            out_dtype=torch.bfloat16,
        )

    # Shape sanity check.
    if bf16_mm().shape != fp8_mm().shape:
        raise RuntimeError('BF16 and FP8 outputs have different shapes')

    bf16_mean, bf16_p50 = time_cuda(bf16_mm, warmup=warmup, iters=iters)
    fp8_mean, fp8_p50 = time_cuda(fp8_mm, warmup=warmup, iters=iters)
    print(
        f'| {shape.name} | {shape.m} | {shape.k} | {shape.n} | '
        f'{bf16_mean:.4f} | {fp8_mean:.4f} | {bf16_mean / fp8_mean:.2f}x | '
        f'{tflops(shape, bf16_mean):.2f} | {tflops(shape, fp8_mean):.2f} | '
        f'{bf16_p50:.4f} | {fp8_p50:.4f} |'
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if not hasattr(torch, 'float8_e4m3fn') or not hasattr(torch, '_scaled_mm'):
        raise RuntimeError('torch.float8_e4m3fn and torch._scaled_mm are required')

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    shapes = [parse_shape(item) for item in args.shape] if args.shape else DEFAULT_SHAPES

    print(f'device: {device} ({torch.cuda.get_device_name(device)})')
    print('This is GEMM-only timing: FP8 tensors and scales are prepared before timing.')
    print('| shape | M | K | N | bf16 mean ms | fp8 mean ms | speedup | bf16 TFLOP/s | fp8 TFLOP/s | bf16 p50 ms | fp8 p50 ms |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for shape in shapes:
        run_one(shape, device=device, warmup=args.warmup, iters=args.iters)


if __name__ == '__main__':
    main()
