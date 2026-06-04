#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import torch


def parse_gpu_ids(value: str) -> list[int]:
    ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("cuda:"):
            part = part.split(":", 1)[1]
        ids.append(int(part))
    if not ids:
        raise argparse.ArgumentTypeError("at least one GPU id is required")
    return ids


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def allocate_on_gpu(
    gpu_id: int,
    *,
    memory_fraction: float,
    memory_mb: int | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    torch.cuda.set_device(gpu_id)
    free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
    if memory_mb is None:
        target_bytes = int(free_bytes * memory_fraction)
    else:
        target_bytes = int(memory_mb * 1024 * 1024)

    if target_bytes <= 0:
        raise ValueError(f"target allocation must be positive, got {target_bytes} bytes")
    if target_bytes >= free_bytes:
        target_bytes = int(free_bytes * 0.95)

    elem_size = torch.empty((), dtype=dtype, device=f"cuda:{gpu_id}").element_size()
    numel = max(target_bytes // elem_size, 1)
    tensor = torch.empty(numel, dtype=dtype, device=f"cuda:{gpu_id}")
    tensor.fill_(1)
    used_gb = tensor.numel() * elem_size / 1024**3
    free_gb = free_bytes / 1024**3
    total_gb = total_bytes / 1024**3
    print(
        f"cuda:{gpu_id} allocated {used_gb:.2f} GiB "
        f"(free before {free_gb:.2f} GiB / total {total_gb:.2f} GiB)",
        flush=True,
    )
    return tensor


def main() -> None:
    parser = argparse.ArgumentParser(description="Occupy selected GPU memory until interrupted.")
    parser.add_argument("gpus", type=parse_gpu_ids, help='GPU ids, e.g. "0" or "0,1,6".')
    parser.add_argument("--memory-fraction", type=float, default=0.90, help="Fraction of currently free memory to occupy.")
    parser.add_argument("--memory-mb", type=int, default=None, help="Fixed memory per GPU in MiB. Overrides --memory-fraction.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--sleep-seconds", type=float, default=60.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if args.memory_fraction <= 0 or args.memory_fraction > 1:
        raise ValueError("--memory-fraction must be in (0, 1].")

    dtype = dtype_from_name(args.dtype)
    tensors = [
        allocate_on_gpu(
            gpu_id,
            memory_fraction=args.memory_fraction,
            memory_mb=args.memory_mb,
            dtype=dtype,
        )
        for gpu_id in args.gpus
    ]

    print("GPU memory occupied. Press Ctrl+C to release.", flush=True)
    try:
        while True:
            time.sleep(args.sleep_seconds)
            # Keep references alive and touch one value so aggressive optimizers cannot discard them.
            for tensor in tensors:
                _ = tensor[0].item()
    except KeyboardInterrupt:
        print("Releasing GPU memory.", flush=True)


if __name__ == "__main__":
    main()
