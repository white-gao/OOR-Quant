"""Measure actual FP8 activation-QDQ error at the Stage-A prefill position.

This is an analysis-only runner.  It uses the ordinary real GPTQ W8A8 model,
patches ``RealFP8Linear.prepare_input`` to observe its inputs, and preserves
the model's numerical output exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from real_quant.full_precision.run_hf_baseline import (
    DEFAULT_DATA_DIR,
    load_task_data,
    parse_sample_size,
    resolve_repo_path,
)

from .apply import NaiveW8A8Summary
from .gptq_runtime import build_model_batches, get_transformer_layers, parse_layer_indices
from .modules import FP8PreparedInput, RealFP8Linear, require_fp8_runtime
from .run_stage_a_weight_attribution import apply_gptq_with_weight_snapshots


DEFAULT_OUTPUT_DIR = "real_quant/naive_w8a8/results/probes/stage_a_activation_error"
TOP_FRACTIONS = (0.01, 0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe real FP8 Stage-A activation-QDQ channel error.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--task", default="ad")
    parser.add_argument("--calib_split", default="calib")
    parser.add_argument("--calib_sample_size", default="128")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--slot_profile",
        default="fake_quant_learnable/results/analysis/slot_tokenwise_outlier_channels/"
        "ad_OneRec-1.7B_s128_all_layers_top1pct/tokenwise_channel_frequency.pt",
    )
    return parser.parse_args()


def _module_label(module: RealFP8Linear) -> str:
    name = getattr(module, "stage_a_probe_name", "")
    if ".self_attn.q_proj" in name:
        return "qkv_input"
    if ".mlp.gate_proj" in name:
        return "gate_up_input"
    if ".self_attn.o_proj" in name:
        return "o_proj"
    if ".mlp.down_proj" in name:
        return "down_proj"
    return name.rsplit(".", 1)[-1]


def _top_indices(values: torch.Tensor, fraction: float) -> torch.Tensor:
    count = max(1, int(round(values.numel() * fraction)))
    return torch.topk(values, k=count, largest=True).indices


@contextmanager
def capture_stage_a_qdq_error(model: nn.Module) -> Iterator[dict[str, dict[str, torch.Tensor | int]]]:
    """Accumulate final-prefill-token input error for every distinct Linear path."""
    modules = list(model.named_modules())
    for name, module in modules:
        if isinstance(module, RealFP8Linear):
            module.stage_a_probe_name = name

    original_prepare = RealFP8Linear.prepare_input
    stats: dict[str, dict[str, torch.Tensor | int]] = {}

    def patched_prepare(module: RealFP8Linear, x: torch.Tensor) -> FP8PreparedInput:
        prepared = original_prepare(module, x)
        # ``prepare_input`` is called once for shared QKV/gate-up inputs too.
        # In prefill, its last row is exactly the Stage-A token at this Linear.
        if x.ndim >= 3 and int(x.shape[-2]) > 1:
            source = x.detach().float().reshape(-1, module.in_features)[-1]
            qdq = (prepared.x_fp8.float() * prepared.scale.float())[-1]
            error2 = (source - qdq).square().cpu()
            source2 = source.square().cpu()
            label = _module_label(module)
            key = f"layer{getattr(module, 'stage_a_probe_name').split('.layers.')[1].split('.')[0]}.{label}"
            entry = stats.setdefault(key, {"error2": torch.zeros_like(error2), "source2": torch.zeros_like(source2), "count": 0})
            entry["error2"] = entry["error2"] + error2
            entry["source2"] = entry["source2"] + source2
            entry["count"] = int(entry["count"]) + 1
        return prepared

    RealFP8Linear.prepare_input = patched_prepare  # type: ignore[method-assign]
    try:
        yield stats
    finally:
        RealFP8Linear.prepare_input = original_prepare  # type: ignore[method-assign]
        for _name, module in modules:
            if isinstance(module, RealFP8Linear) and hasattr(module, "stage_a_probe_name"):
                delattr(module, "stage_a_probe_name")


def _load_slot_profiles(path: Path) -> Mapping[str, torch.Tensor]:
    if not path.exists():
        return {}
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): value for key, value in raw.items() if torch.is_tensor(value)}


def summarize(
    stats: Mapping[str, Mapping[str, torch.Tensor | int]],
    slot_profiles: Mapping[str, torch.Tensor],
) -> tuple[list[dict[str, object]], dict[str, torch.Tensor]]:
    rows: list[dict[str, object]] = []
    vectors: dict[str, torch.Tensor] = {}
    for key, entry in sorted(stats.items()):
        count = int(entry["count"])
        error2 = entry["error2"] / max(1, count)  # type: ignore[operator]
        source2 = entry["source2"] / max(1, count)  # type: ignore[operator]
        relative = error2 / source2.clamp_min(1e-12)
        vectors[f"{key}.mean_error2"] = error2
        vectors[f"{key}.relative_error2"] = relative
        row: dict[str, object] = {
            "path": key,
            "samples": count,
            "channels": error2.numel(),
            "mean_mse": float(error2.mean().item()),
            "mean_relative_mse": float(relative.mean().item()),
        }
        for fraction in TOP_FRACTIONS:
            indices = _top_indices(error2, fraction)
            tag = f"top_{int(fraction * 100)}pct"
            row[f"{tag}_error_mass"] = float(error2[indices].sum().item() / error2.sum().clamp_min(1e-12).item())
            row[f"{tag}_channels"] = ";".join(map(str, indices.tolist()))

        # This is an alignment diagnostic, not a causal comparison: previous
        # slot profiles came from BF16 prompt activations, while ``error2`` is
        # collected from the real W8A8 propagated Stage-A path.
        layer, path_name = key.split(".", 1)
        source_name = {"qkv_input": "self_attn.q_proj", "gate_up_input": "mlp.gate_proj", "o_proj": "self_attn.o_proj", "down_proj": "mlp.down_proj"}.get(path_name)
        if source_name is not None:
            for group in ("sid_a", "sid_b", "sid_c", "text", "boundary"):
                profile_key = f"{layer}.{source_name}.channel_frequency.{group}"
                frequency = slot_profiles.get(profile_key)
                if frequency is None or frequency.numel() != error2.numel():
                    continue
                group_top = _top_indices(frequency.float(), 0.01)
                row[f"slot_{group}_top1pct_error_mass"] = float(error2[group_top].sum().item() / error2.sum().clamp_min(1e-12).item())
        rows.append(row)
    return rows, vectors


def main() -> None:
    args = parse_args()
    require_fp8_runtime()
    dtype = torch.bfloat16 if args.dtype in {"bf16", "bfloat16"} else torch.float16
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=dtype).to(args.device)
    model.eval()
    from benchmark.tasks.v1_0.registry import get_task_config

    config = get_task_config(args.task)
    prompt_token = config.get("generation_config", {}).get("prompt_token", "<|sid_begin|>")
    calib_data = load_task_data(task_name=args.task, data_dir=str(resolve_repo_path(args.data_dir)), tokenizer=tokenizer, split=args.calib_split, sample_size=parse_sample_size(args.calib_sample_size))
    prompts = [sample["prompt"] + prompt_token if not sample["prompt"].endswith(prompt_token) else sample["prompt"] for sample in calib_data.values()]
    batches = build_model_batches(tokenizer=tokenizer, prompts=prompts, device=torch.device(args.device))
    summary: NaiveW8A8Summary = apply_gptq_with_weight_snapshots(model=model, model_batches=batches, layer_indices=parse_layer_indices("all", num_layers=len(get_transformer_layers(model))), output_dtype=dtype)
    print(f"[stage_a_activation_error] GPTQ linears={summary.replaced_linears}")
    forward_module = model.model if hasattr(model, "model") else model
    with capture_stage_a_qdq_error(model) as stats, torch.no_grad():
        for index, batch in enumerate(batches):
            forward_module(**batch, use_cache=False)
            if (index + 1) % 16 == 0 or index + 1 == len(batches):
                print(f"[stage_a_activation_error] sample {index + 1}/{len(batches)}")
    rows, vectors = summarize(stats, _load_slot_profiles(resolve_repo_path(args.slot_profile)))
    with (output_dir / "stage_a_activation_error.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    torch.save(vectors, output_dir / "stage_a_activation_error_vectors.pt")
    report = {
        "protocol": {"task": args.task, "calib_split": args.calib_split, "calib_samples": len(batches), "quantization": "real GPTQ W8A8", "position": "prefill final sid_begin", "error": "per-channel mean squared FP8-QDQ input error"},
        "paths": len(rows),
        "rows": rows,
        "caveat": "slot alignment uses previous BF16-prompt token-wise outlier profiles and is descriptive only.",
    }
    (output_dir / "stage_a_activation_error_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output_dir / "stage_a_activation_error_summary.json")


if __name__ == "__main__":
    main()
