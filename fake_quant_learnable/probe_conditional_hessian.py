from __future__ import annotations

"""Probe whether Linear output channels have stable slot-conditional usage.

This is an analysis-only script.  It does not alter quantization code or save a
quantized model.  For a Linear output channel r and prompt slot group g, it
collects the token-count-normalized output energy A[r, g], then defines
pi[r, g] = A[r, g] / sum_h A[r, h].  A non-uniform, stable pi is the necessary
empirical premise for using channel-conditional Hessians during offline GPTQ.
"""

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from fake_quant_learnable.token_weights import SLOT_TOKEN_GROUPS, build_prompt_slot_token_group_batches
from real_quant.full_precision.generator import dtype_from_name
from real_quant.full_precision.run_hf_baseline import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_PATH,
    RECOMMENDATION_TASKS,
    load_task_data,
    parse_sample_size,
    resolve_repo_path,
)
from real_quant.naive_w8a8.gptq_runtime import (
    build_model_batches,
    default_calib_split,
    format_prompt,
    get_transformer_layers,
    parse_layer_indices,
)


DEFAULT_OUTPUT_ROOT = "fake_quant_learnable/results/analysis/conditional_hessian_probe"
DEFAULT_LINEAR_REGEX = r"(q_proj|o_proj|gate_proj|down_proj)$"
EPS = 1e-12


@dataclass
class ConditionalEnergyAccumulator:
    """Streaming balanced-group output-energy statistics for one Linear."""

    num_groups: int
    sum_sq: torch.Tensor | None = None
    counts: torch.Tensor | None = None
    half_sum_sq: list[torch.Tensor | None] = field(default_factory=lambda: [None, None])
    half_counts: list[torch.Tensor | None] = field(default_factory=lambda: [None, None])

    def add(self, values: torch.Tensor, groups: torch.Tensor, valid_mask: torch.Tensor, half: int) -> None:
        if half not in (0, 1):
            raise ValueError(f"Expected split half 0 or 1, got {half}.")
        output = values.detach().float().reshape(-1, values.shape[-1])
        group_flat = groups.to(device=output.device).reshape(-1)
        valid_flat = valid_mask.to(device=output.device).reshape(-1).bool()
        if output.shape[0] != group_flat.numel() or output.shape[0] != valid_flat.numel():
            raise ValueError("Linear output and token-group shapes do not match.")

        # Per-token RMS normalization makes A describe channel preference rather
        # than a group-wide output-scale difference.
        normalized_sq = output.square() / output.square().mean(dim=-1, keepdim=True).clamp_min(EPS)
        if self.sum_sq is None:
            shape = (self.num_groups, output.shape[-1])
            self.sum_sq = torch.zeros(shape, device=output.device, dtype=torch.float32)
            self.counts = torch.zeros(self.num_groups, device=output.device, dtype=torch.long)
            self.half_sum_sq = [torch.zeros_like(self.sum_sq), torch.zeros_like(self.sum_sq)]
            self.half_counts = [torch.zeros_like(self.counts), torch.zeros_like(self.counts)]
        assert self.sum_sq is not None and self.counts is not None
        assert self.half_sum_sq[half] is not None and self.half_counts[half] is not None
        for group_id in range(self.num_groups):
            mask = valid_flat & (group_flat == group_id)
            if not bool(mask.any()):
                continue
            group_sum = normalized_sq[mask].sum(dim=0)
            group_count = int(mask.sum().item())
            self.sum_sq[group_id] += group_sum
            self.counts[group_id] += group_count
            self.half_sum_sq[half][group_id] += group_sum
            self.half_counts[half][group_id] += group_count

    def energy(self, half: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if half is None:
            sums, counts = self.sum_sq, self.counts
        else:
            sums, counts = self.half_sum_sq[half], self.half_counts[half]
        if sums is None or counts is None:
            raise ValueError("No activations were accumulated.")
        return (sums / counts.clamp_min(1).to(sums.dtype).unsqueeze(1)).detach().cpu(), counts.detach().cpu()


@dataclass
class ConditionalHessianProbeResult:
    accumulators: dict[tuple[int, str], ConditionalEnergyAccumulator] = field(default_factory=dict)

    def add(
        self,
        *,
        layer: int,
        module: str,
        values: torch.Tensor,
        groups: torch.Tensor,
        valid_mask: torch.Tensor,
        half: int,
    ) -> None:
        key = (layer, module)
        if key not in self.accumulators:
            self.accumulators[key] = ConditionalEnergyAccumulator(num_groups=len(SLOT_TOKEN_GROUPS))
        self.accumulators[key].add(values, groups, valid_mask, half)


def conditional_mixture(energy: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    """Return pi[channel, group] from balanced group-by-channel energy [G, C]."""
    if energy.ndim != 2 or energy.shape[0] != len(SLOT_TOKEN_GROUPS):
        raise ValueError(f"Expected energy [{len(SLOT_TOKEN_GROUPS)}, channels], got {tuple(energy.shape)}.")
    nonnegative = energy.detach().float().clamp_min(0.0)
    return nonnegative.transpose(0, 1) / nonnegative.sum(dim=0).clamp_min(eps).unsqueeze(1)


def normalized_entropy(pi: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    """Per-channel entropy in [0, 1]; zero means a one-group channel."""
    if pi.ndim != 2 or pi.shape[1] != len(SLOT_TOKEN_GROUPS):
        raise ValueError(f"Expected pi [channels, {len(SLOT_TOKEN_GROUPS)}], got {tuple(pi.shape)}.")
    return -(pi.clamp_min(eps) * pi.clamp_min(eps).log()).sum(dim=1) / math.log(len(SLOT_TOKEN_GROUPS))


def channelwise_cosine(first: torch.Tensor, second: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    if first.shape != second.shape:
        raise ValueError(f"Mismatched pi shapes: {tuple(first.shape)} vs {tuple(second.shape)}")
    return (first * second).sum(dim=1) / (first.norm(dim=1) * second.norm(dim=1)).clamp_min(eps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the premise of slot-conditional Hessians in OneRec.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--task", default="ad", choices=RECOMMENDATION_TASKS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--calib_split", default=None)
    parser.add_argument("--sample_size", default="128")
    parser.add_argument("--layers", default="last:4", help='Layer spec: "all", "last:K", or "0,2-4".')
    parser.add_argument("--linear_regex", default=DEFAULT_LINEAR_REGEX)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt_token", default="<|sid_begin|>")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--progress_every", type=int, default=16)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype_from_name(args.dtype),
    ).to(device)
    model.eval()

    calib_split = args.calib_split or default_calib_split(
        args.data_dir, args.split, task_name=args.task, resolve_path=resolve_repo_path
    )
    calib_data = load_task_data(
        task_name=args.task,
        data_dir=str(resolve_repo_path(args.data_dir)),
        tokenizer=tokenizer,
        split=calib_split,
        sample_size=parse_sample_size(args.sample_size),
    )
    samples = list(calib_data.values())
    prompts = [format_prompt(sample["prompt"], args.prompt_token) for sample in samples]
    batches = build_model_batches(tokenizer=tokenizer, prompts=prompts, device=device)
    groups = build_prompt_slot_token_group_batches(tokenizer=tokenizer, prompts=prompts, device=device)
    layers = get_transformer_layers(model)
    layer_indices = parse_layer_indices(args.layers, num_layers=len(layers))
    output_dir = _resolve_output_dir(args, sample_count=len(samples), layer_indices=layer_indices)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[probe_conditional_hessian] collecting Linear-output slot mixtures "
        f"task={args.task}, split={calib_split}, samples={len(samples)}, layers={layer_indices}"
    )
    result = collect_conditional_hessian_profiles(
        model=model,
        layers=layers,
        layer_indices=layer_indices,
        model_batches=batches,
        slot_group_batches=groups,
        linear_regex=args.linear_regex,
        progress_every=args.progress_every,
    )
    channel_rows, summary_rows, tensors = build_probe_outputs(result)
    _write_csv(output_dir / "channel_slot_mixtures.csv", channel_rows)
    _write_csv(output_dir / "linear_conditionality_summary.csv", summary_rows)
    torch.save(tensors, output_dir / "conditional_hessian_profiles.pt")
    summary = {
        "task": args.task,
        "split": calib_split,
        "sample_size": len(samples),
        "model_path": args.model_path,
        "layers": layer_indices,
        "linear_regex": args.linear_regex,
        "slot_groups": list(SLOT_TOKEN_GROUPS),
        "energy_definition": "A[g,r] = mean_{t in g}((Y[t,r]^2) / mean_c(Y[t,c]^2))",
        "mixture_definition": "pi[r,g] = A[g,r] / sum_h A[h,r]",
        "output_files": {
            "channel_mixtures": "channel_slot_mixtures.csv",
            "summary": "linear_conditionality_summary.csv",
            "tensors": "conditional_hessian_profiles.pt",
            "report": "report.md",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(summary, summary_rows), encoding="utf-8")
    print(f"[probe_conditional_hessian] saved to {output_dir}")


def collect_conditional_hessian_profiles(
    *,
    model: nn.Module,
    layers: Sequence[nn.Module],
    layer_indices: Sequence[int],
    model_batches: Sequence[Mapping[str, Any]],
    slot_group_batches: Sequence[torch.Tensor],
    linear_regex: str = DEFAULT_LINEAR_REGEX,
    progress_every: int = 16,
) -> ConditionalHessianProbeResult:
    if len(model_batches) != len(slot_group_batches):
        raise ValueError("slot_group_batches length does not match model_batches length.")
    selected = sorted(set(int(index) for index in layer_indices))
    pattern = re.compile(linear_regex)
    result = ConditionalHessianProbeResult()
    state: dict[str, torch.Tensor | int | None] = {"groups": None, "valid_mask": None, "half": None, "captured": 0}
    handles: list[Any] = []

    def make_hook(layer_idx: int, module_name: str):
        def hook(_module: nn.Module, _args: tuple[Any, ...], _kwargs: dict[str, Any], output: Any) -> None:
            values = output[0] if isinstance(output, tuple) else output
            groups = state["groups"]
            valid_mask = state["valid_mask"]
            half = state["half"]
            if not torch.is_tensor(values) or not torch.is_tensor(groups) or not torch.is_tensor(valid_mask):
                return
            if not isinstance(half, int):
                raise RuntimeError("Missing split-half state while collecting outputs.")
            result.add(
                layer=layer_idx,
                module=module_name,
                values=values,
                groups=groups,
                valid_mask=valid_mask,
                half=half,
            )
            state["captured"] = int(state["captured"] or 0) + 1

        return hook

    for layer_idx in selected:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of range for {len(layers)} layers.")
        for module_name, module in layers[layer_idx].named_modules():
            if isinstance(module, nn.Linear) and pattern.search(module_name):
                handles.append(module.register_forward_hook(make_hook(layer_idx, module_name), with_kwargs=True))
    if not handles:
        raise ValueError(f"No Linear modules matched regex {linear_regex!r} in layers {selected}.")

    was_training = model.training
    forward_module = model.model if hasattr(model, "model") and hasattr(model.model, "layers") else model
    try:
        with torch.no_grad():
            for batch_idx, (batch, group_ids) in enumerate(zip(model_batches, slot_group_batches)):
                state["groups"] = group_ids
                attention_mask = batch.get("attention_mask")
                state["valid_mask"] = attention_mask.bool() if torch.is_tensor(attention_mask) else torch.ones_like(group_ids, dtype=torch.bool)
                state["half"] = batch_idx % 2
                state["captured"] = 0
                try:
                    forward_module(**batch, use_cache=False)
                except TypeError:
                    forward_module(**batch)
                if int(state["captured"] or 0) == 0:
                    raise RuntimeError("No Linear outputs were captured during model forward.")
                if progress_every > 0 and ((batch_idx + 1) % progress_every == 0 or batch_idx + 1 == len(model_batches)):
                    print(f"[probe_conditional_hessian] sample {batch_idx + 1}/{len(model_batches)} captured_modules={state['captured']}")
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return result


def build_probe_outputs(
    result: ConditionalHessianProbeResult,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, torch.Tensor]]:
    channel_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    tensors: dict[str, torch.Tensor] = {}
    for (layer, module), accumulator in sorted(result.accumulators.items()):
        energy, counts = accumulator.energy()
        if bool((counts <= 0).any()):
            missing = [SLOT_TOKEN_GROUPS[index] for index, count in enumerate(counts.tolist()) if count <= 0]
            raise ValueError(f"Missing token groups for layer={layer}, module={module}: {missing}")
        pi = conditional_mixture(energy)
        entropy = normalized_entropy(pi)
        half_pi = [conditional_mixture(accumulator.energy(half)[0]) for half in (0, 1)]
        half_cosine = channelwise_cosine(half_pi[0], half_pi[1])
        half_top1_agree = half_pi[0].argmax(dim=1).eq(half_pi[1].argmax(dim=1)).float()
        dominant = pi.argmax(dim=1)
        maximum = pi.max(dim=1).values
        tensors[f"layer{layer}.{module}.energy"] = energy
        tensors[f"layer{layer}.{module}.pi"] = pi
        tensors[f"layer{layer}.{module}.half0_pi"] = half_pi[0]
        tensors[f"layer{layer}.{module}.half1_pi"] = half_pi[1]

        for channel in range(pi.shape[0]):
            row: dict[str, Any] = {
                "layer": layer,
                "module": module,
                "channel": channel,
                "entropy": _round(float(entropy[channel].item())),
                "max_pi": _round(float(maximum[channel].item())),
                "dominant_group": SLOT_TOKEN_GROUPS[int(dominant[channel].item())],
                "split_half_cosine": _round(float(half_cosine[channel].item())),
                "split_half_top1_agree": bool(half_top1_agree[channel].item()),
            }
            for group_id, group in enumerate(SLOT_TOKEN_GROUPS):
                row[f"pi_{group}"] = _round(float(pi[channel, group_id].item()))
            channel_rows.append(row)

        summary: dict[str, Any] = {
            "layer": layer,
            "module": module,
            "output_channels": int(pi.shape[0]),
            "mean_entropy": _round(float(entropy.mean().item())),
            "median_entropy": _round(float(entropy.median().item())),
            "mean_max_pi": _round(float(maximum.mean().item())),
            "fraction_entropy_le_0.8": _round(float((entropy <= 0.8).float().mean().item())),
            "fraction_max_pi_ge_0.35": _round(float((maximum >= 0.35).float().mean().item())),
            "fraction_max_pi_ge_0.5": _round(float((maximum >= 0.5).float().mean().item())),
            "mean_split_half_cosine": _round(float(half_cosine.mean().item())),
            "split_half_top1_agreement": _round(float(half_top1_agree.mean().item())),
        }
        for group_id, group in enumerate(SLOT_TOKEN_GROUPS):
            summary[f"dominant_{group}_fraction"] = _round(float((dominant == group_id).float().mean().item()))
            summary[f"token_count_{group}"] = int(counts[group_id].item())
        summary_rows.append(summary)
    return channel_rows, summary_rows, tensors


def build_report(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda row: float(row["mean_entropy"]))
    lines = [
        "# Conditional Hessian Premise Probe",
        "",
        f"- task: `{summary['task']}`",
        f"- split: `{summary['split']}`",
        f"- samples: `{summary['sample_size']}`",
        f"- layers: `{summary['layers']}`",
        f"- linears: `{summary['linear_regex']}`",
        "",
        "## Definition",
        "",
        "```text",
        "A[g,r] = mean_{t in g}(Y[t,r]^2 / mean_c(Y[t,c]^2))",
        "pi[r,g] = A[g,r] / sum_h A[h,r]",
        "H_r = sum_g pi[r,g] H_g",
        "```",
        "",
        "`pi[r,:]` is computed with equal group priors. Entropy 1.0 and max-pi 0.2 indicate no slot conditioning.",
        "",
        "## Most Conditional Linear Outputs",
        "",
        "| layer | module | mean entropy | mean max-pi | max-pi >= 0.35 | split-half cosine | top-1 agreement |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered[:16]:
        lines.append(
            f"| {row['layer']} | {row['module']} | {float(row['mean_entropy']):.4f} | "
            f"{float(row['mean_max_pi']):.4f} | {float(row['fraction_max_pi_ge_0.35']):.4f} | "
            f"{float(row['mean_split_half_cosine']):.4f} | {float(row['split_half_top1_agreement']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "Proceed only if slot mixtures are both non-uniform (entropy materially below 1.0 / max-pi above 0.2) and stable across halves.",
            "The CSV contains per-channel mixtures; the PT file contains the complete matrices for later clustering.",
        ]
    )
    return "\n".join(lines) + "\n"


def _resolve_output_dir(args: argparse.Namespace, *, sample_count: int, layer_indices: Sequence[int]) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    layer_tag = "all" if len(layer_indices) >= 28 else f"l{min(layer_indices)}-{max(layer_indices)}"
    return Path(DEFAULT_OUTPUT_ROOT) / f"{args.task}_1p7b_s{sample_count}_{layer_tag}"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _round(value: float) -> float:
    return round(value, 8)


if __name__ == "__main__":
    main()
