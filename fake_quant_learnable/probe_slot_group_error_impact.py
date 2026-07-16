#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from fake_quant_learnable.apply import install_shared_input_activation_quantization
from fake_quant_learnable.gptq import (
    DEFAULT_GPTQ_BLOCK_SIZE,
    DEFAULT_GPTQ_DAMP_PERCENT,
    collect_gptq_hessians,
    gptq_quantized_module_from_hessians,
)
from fake_quant_learnable.gradient_weights import (
    _build_teacher_forcing_batch,
    _extract_logits,
    _forward_model,
    _teacher_forcing_full_sid_loss,
)
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
    advance_layer_input_batches,
    build_model_batches,
    build_sid_teacher_forcing_target_token_ids,
    capture_layer_input_batches,
    default_calib_split,
    format_prompt,
    get_transformer_layers,
    parse_layer_indices,
)


DEFAULT_OUTPUT_ROOT = "fake_quant_learnable/results/analysis/slot_group_error_impact"
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe whether minority SID groups have high reconstruction error and end-loss impact."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--task", default="ad", choices=RECOMMENDATION_TASKS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--calib_split", default=None)
    parser.add_argument("--calib_sample_size", default="128")
    parser.add_argument("--probe_sample_size", default="32")
    parser.add_argument("--layers", default="last:5", help='Layer spec: "all", "last:K", or "0,2-4".')
    parser.add_argument("--max_targets", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt_token", default="<|sid_begin|>")
    parser.add_argument("--damp_percent", type=float, default=DEFAULT_GPTQ_DAMP_PERCENT)
    parser.add_argument("--block_size", type=int, default=DEFAULT_GPTQ_BLOCK_SIZE)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--progress_every", type=int, default=1)
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
    layers = get_transformer_layers(model)
    layer_indices = parse_layer_indices(args.layers, num_layers=len(layers))
    calib_split = args.calib_split or default_calib_split(
        args.data_dir, args.split, task_name=args.task, resolve_path=resolve_repo_path
    )
    calib_samples = _load_samples(
        task=args.task,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        split=calib_split,
        sample_size=args.calib_sample_size,
    )
    probe_samples = _load_samples(
        task=args.task,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        split=args.split,
        sample_size=args.probe_sample_size,
    )
    calib_prompts = [format_prompt(sample["prompt"], args.prompt_token) for sample in calib_samples]
    probe_prompts = [format_prompt(sample["prompt"], args.prompt_token) for sample in probe_samples]
    calib_batches = build_model_batches(tokenizer=tokenizer, prompts=calib_prompts, device=device)
    probe_batches = build_model_batches(tokenizer=tokenizer, prompts=probe_prompts, device=device)
    probe_groups = build_prompt_slot_token_group_batches(tokenizer=tokenizer, prompts=probe_prompts, device=device)
    target_sequences = build_sid_teacher_forcing_target_token_ids(
        tokenizer, probe_samples, max_items=args.max_targets
    )
    output_dir = _resolve_output_dir(args, layer_indices=layer_indices)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[slot_group_error_impact] "
        f"calibrating plain GPTQ blocks: calib={len(calib_batches)}, probe={len(probe_batches)}, "
        f"layers={layer_indices}"
    )
    rows: list[dict[str, Any]] = []
    stream_inputs = None
    stream_layer_idx = None
    for ordinal, layer_idx in enumerate(layer_indices):
        if stream_inputs is None:
            stream_inputs = capture_layer_input_batches(
                model=model, layer=layers[layer_idx], model_batches=calib_batches
            )
            stream_layer_idx = layer_idx
        else:
            assert stream_layer_idx is not None
            while stream_layer_idx < layer_idx:
                stream_inputs = advance_layer_input_batches(layer=layers[stream_layer_idx], batches=stream_inputs)
                stream_layer_idx += 1

        teacher_block = layers[layer_idx]
        hessians = collect_gptq_hessians(teacher_block, stream_inputs)
        quant_block, _ = gptq_quantized_module_from_hessians(
            copy.deepcopy(teacher_block),
            hessians,
            act_quant="per_token",
            damp_percent=args.damp_percent,
            block_size=args.block_size,
        )
        quant_block.to(device).eval()
        install_shared_input_activation_quantization(quant_block)
        print(f"[slot_group_error_impact] probing layer={layer_idx} ({ordinal + 1}/{len(layer_indices)})")
        rows.extend(
            probe_layer(
                model=model,
                layer=teacher_block,
                quant_block=quant_block,
                layer_idx=layer_idx,
                probe_batches=probe_batches,
                slot_group_batches=probe_groups,
                target_sequences=target_sequences,
                progress_every=args.progress_every,
            )
        )
        del quant_block, hessians
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        stream_inputs = advance_layer_input_batches(layer=teacher_block, batches=stream_inputs)
        stream_layer_idx = layer_idx + 1

    summary_rows = aggregate_rows(rows)
    _write_csv(output_dir / "sample_group_error_impact.csv", rows)
    _write_csv(output_dir / "layer_group_summary.csv", summary_rows)
    report = build_report(args=args, calib_split=calib_split, layer_indices=layer_indices, summary_rows=summary_rows)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "task": args.task,
                "split": calib_split,
                "calib_sample_size": len(calib_samples),
                "probe_sample_size": len(probe_samples),
                "layers": layer_indices,
                "method": "plain_gptq_w8a8_block_residual_injection",
                "groups": list(SLOT_TOKEN_GROUPS),
                "files": {
                    "sample_rows": "sample_group_error_impact.csv",
                    "summary": "layer_group_summary.csv",
                    "report": "report.md",
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[slot_group_error_impact] saved to {output_dir}")


def _load_samples(*, task: str, data_dir: str, tokenizer: Any, split: str, sample_size: str) -> list[Mapping[str, Any]]:
    data = load_task_data(
        task_name=task,
        data_dir=str(resolve_repo_path(data_dir)),
        tokenizer=tokenizer,
        split=split,
        sample_size=parse_sample_size(sample_size),
    )
    return list(data.values())


def probe_layer(
    *,
    model: nn.Module,
    layer: nn.Module,
    quant_block: nn.Module,
    layer_idx: int,
    probe_batches: Sequence[Mapping[str, torch.Tensor]],
    slot_group_batches: Sequence[torch.Tensor],
    target_sequences: Sequence[Sequence[torch.Tensor]],
    progress_every: int,
) -> list[dict[str, Any]]:
    if not (len(probe_batches) == len(slot_group_batches) == len(target_sequences)):
        raise ValueError("Probe batch, group, and target sequence counts must match.")
    state: dict[str, Any] = {"mode": "capture", "groups": None, "prompt_len": None, "captured": None}

    def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any):
        hidden = _output_hidden(output)
        if state["mode"] == "capture":
            with torch.no_grad():
                q_hidden = _output_hidden(quant_block(*args, **kwargs))
            state["captured"] = (hidden.detach(), q_hidden.detach())
            return None
        if state["mode"] != "inject":
            return None
        groups = state["groups"]
        prompt_len = state["prompt_len"]
        if not torch.is_tensor(groups) or not isinstance(prompt_len, int):
            raise RuntimeError("Injection state is incomplete.")
        with torch.no_grad():
            q_hidden = _output_hidden(quant_block(*args, **kwargs))
        group_mask = groups.to(device=hidden.device).eq(int(state["group_id"]))
        group_mask = group_mask[:, :prompt_len]
        full_mask = torch.zeros(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        full_mask[:, :prompt_len] = group_mask
        injected = torch.where(full_mask.unsqueeze(-1), q_hidden, hidden)
        return _replace_output_hidden(output, injected)

    handle = layer.register_forward_hook(hook, with_kwargs=True)
    rows: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for sample_idx, (batch, groups, targets) in enumerate(zip(probe_batches, slot_group_batches, target_sequences)):
                teacher_batch, prompt_mask, prompt_len, loss_specs = _build_teacher_forcing_batch(batch, targets)
                state["mode"] = "capture"
                state["captured"] = None
                base_loss = float(_teacher_forcing_full_sid_loss(_extract_logits(_forward_model(model, teacher_batch)), loss_specs).item())
                captured = state["captured"]
                if captured is None:
                    raise RuntimeError(f"Layer {layer_idx} did not produce a captured output.")
                fp_hidden, q_hidden = captured
                group_prompt = _prompt_groups(
                    groups,
                    prompt_mask=prompt_mask,
                    prompt_len=prompt_len,
                    batch_size=fp_hidden.shape[0],
                )
                for group_id, group_name in enumerate(SLOT_TOKEN_GROUPS):
                    mask = group_prompt.eq(group_id)
                    count = int(mask.sum().item())
                    if count == 0:
                        continue
                    fp_values = fp_hidden[:, :prompt_len, :][mask]
                    q_values = q_hidden[:, :prompt_len, :][mask]
                    sq_error = float((q_values.float() - fp_values.float()).square().sum().item())
                    signal = float(fp_values.float().square().sum().item())
                    relative_error = sq_error / max(signal, EPS)
                    state.update(
                        {
                            "mode": "inject",
                            "groups": group_prompt,
                            "prompt_len": prompt_len,
                            "group_id": group_id,
                        }
                    )
                    injected_loss = float(
                        _teacher_forcing_full_sid_loss(_extract_logits(_forward_model(model, teacher_batch)), loss_specs).item()
                    )
                    rows.append(
                        {
                            "layer": layer_idx,
                            "sample": sample_idx,
                            "group": group_name,
                            "token_count": count,
                            "relative_reconstruction_error": relative_error,
                            "mean_squared_error": sq_error / max(count * fp_hidden.shape[-1], 1),
                            "base_full_sid_ce": base_loss,
                            "injected_full_sid_ce": injected_loss,
                            "ce_delta": injected_loss - base_loss,
                            "ce_delta_per_relative_error": (injected_loss - base_loss) / max(relative_error, EPS),
                        }
                    )
                if progress_every > 0 and ((sample_idx + 1) % progress_every == 0 or sample_idx + 1 == len(probe_batches)):
                    print(f"[slot_group_error_impact] layer={layer_idx} sample={sample_idx + 1}/{len(probe_batches)}")
    finally:
        handle.remove()
    return rows


def _prompt_groups(
    groups: torch.Tensor,
    *,
    prompt_mask: torch.Tensor,
    prompt_len: int,
    batch_size: int,
) -> torch.Tensor:
    valid_groups = groups[prompt_mask.to(device=groups.device)].reshape(-1)
    if valid_groups.numel() != prompt_len:
        raise ValueError(f"Expected {prompt_len} prompt group ids, got {valid_groups.numel()}.")
    return valid_groups.unsqueeze(0).expand(batch_size, -1)


def _output_hidden(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported transformer layer output type: {type(output)!r}")


def _replace_output_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    raise TypeError(f"Unsupported transformer layer output type: {type(output)!r}")


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), str(row["group"]))].append(row)
    result = []
    for (layer, group), items in sorted(grouped.items()):
        result.append(
            {
                "layer": layer,
                "group": group,
                "samples": len(items),
                "mean_token_count": _mean(items, "token_count"),
                "mean_relative_reconstruction_error": _mean(items, "relative_reconstruction_error"),
                "mean_squared_error": _mean(items, "mean_squared_error"),
                "mean_ce_delta": _mean(items, "ce_delta"),
                "positive_ce_delta_fraction": _mean_bool(items, "ce_delta"),
                "mean_ce_delta_per_relative_error": _mean(items, "ce_delta_per_relative_error"),
            }
        )
    total_tokens_by_layer: dict[int, float] = defaultdict(float)
    for row in result:
        total_tokens_by_layer[int(row["layer"])] += float(row["mean_token_count"])
    for row in result:
        total = total_tokens_by_layer[int(row["layer"])]
        frequency = float(row["mean_token_count"]) / max(total, EPS)
        row["mean_token_fraction"] = frequency
        row["global_mse_contribution_proxy"] = frequency * float(row["mean_relative_reconstruction_error"])
    return result


def _mean(items: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(sum(float(item[key]) for item in items) / max(len(items), 1))


def _mean_bool(items: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(sum(float(item[key]) > 0.0 for item in items) / max(len(items), 1))


def build_report(*, args: argparse.Namespace, calib_split: str, layer_indices: Sequence[int], summary_rows: Sequence[Mapping[str, Any]]) -> str:
    rows_by_layer: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        rows_by_layer[int(row["layer"])].append(row)
    lines = [
        "# Slot Group Error-Impact Probe",
        "",
        "This probe quantizes one Transformer block at a time with plain GPTQ W8A8. For each prompt group, it measures the block-output reconstruction error and injects only that group's true quantization residual into the BF16 model before evaluating full-SID teacher-forcing CE.",
        "",
        f"- Calibration: `{args.calib_sample_size}` samples from `{calib_split}`",
        f"- Probe: `{args.probe_sample_size}` samples, full-SID multi-target CE with up to `{args.max_targets}` targets",
        f"- Layers: `{list(layer_indices)}`",
        "",
        "## Layer Summary",
        "",
        "| Layer | Group | Rel. reconstruction error | CE delta after residual injection | Tokens/sample |",
        "|---:|---|---:|---:|---:|",
    ]
    for layer in sorted(rows_by_layer):
        for row in sorted(rows_by_layer[layer], key=lambda item: SLOT_TOKEN_GROUPS.index(str(item["group"]))):
            lines.append(
                f"| {layer} | {row['group']} | {row['mean_relative_reconstruction_error']:.6g} | "
                f"{row['mean_ce_delta']:.6g} | {row['mean_token_count']:.2f} |"
            )
    lines.extend(
        [
            "",
            "Interpretation: a low-frequency SID group with high relative reconstruction error and positive CE delta is evidence that frequency-weighted global reconstruction can under-protect that group.",
        ]
    )
    return "\n".join(lines) + "\n"


def _resolve_output_dir(args: argparse.Namespace, *, layer_indices: Sequence[int]) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    layer_tag = "-".join(str(index) for index in layer_indices)
    return Path(DEFAULT_OUTPUT_ROOT) / f"{args.task}_s{args.probe_sample_size}_c{args.calib_sample_size}_layers{layer_tag}"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows produced for {path.name}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
