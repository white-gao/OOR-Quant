#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from fake_quant.probes.token_sensitivity.probe_sid_token_sensitivity import (
    CHAT_SPECIAL_TOKENS,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_PATH,
    SID_BOUNDARY_TOKENS,
    SID_CODE_RE,
    ActivationCapture,
    dtype_from_name,
    first_sid_target,
    format_prompt,
    install_embedding_grad_anchor,
    load_ad_samples,
    parse_layer_indices,
    parse_str_list,
    register_hooks,
    resolve_repo_path,
    safe_token_text,
    set_seed,
)
from benchmark.tasks.v1_0.registry import get_task_config


DEFAULT_OUTPUT_DIR = "fake_quant_learnable/results/analysis/token_sensitivity/prefill_sample0_s_a_loss"
DEFAULT_LAYERS = "0,8,16,24,27"
DEFAULT_NODES = "attn_qkv_input,attn_o_input,ffn_gate_up_input,ffn_down_input,block_output"
GROUP_COLORS = {
    "text_prompt": "#7f7f7f",
    "history_sid_a": "#2ca02c",
    "history_sid_b": "#1f9d55",
    "history_sid_c": "#98df8a",
    "history_sid_boundary": "#ff7f0e",
    "predict_s_a_position": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-token prefill sensitivity for predicting the first OneRec SID token."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="calib", choices=["calib", "test"])
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--nodes", default=DEFAULT_NODES)
    parser.add_argument("--max_prompt_tokens", type=int, default=0, help="Left-truncate prompt tokens; 0 disables truncation.")
    parser.add_argument("--yscale", default="log", choices=["linear", "log"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def label_prefill_tokens(token_texts: Sequence[str]) -> list[str | None]:
    final_sid_begin = None
    if token_texts and token_texts[-1] == "<|sid_begin|>":
        final_sid_begin = len(token_texts) - 1

    groups: list[str | None] = []
    for idx, token_text in enumerate(token_texts):
        if idx == final_sid_begin:
            groups.append("predict_s_a_position")
            continue
        if token_text in CHAT_SPECIAL_TOKENS:
            groups.append(None)
            continue
        if token_text in SID_BOUNDARY_TOKENS:
            groups.append("history_sid_boundary")
            continue
        sid_match = SID_CODE_RE.match(token_text)
        if sid_match:
            groups.append(f"history_sid_{sid_match.group(1)}")
            continue
        if token_text.startswith("<|"):
            groups.append(None)
            continue
        groups.append("text_prompt")
    return groups


def compute_token_sensitivity(activation: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    if activation.ndim >= 3:
        activation = activation[0]
    if grad.ndim >= 3:
        grad = grad[0]
    if activation.ndim != 2 or grad.ndim != 2:
        raise ValueError(f"Expected [seq, hidden] tensors, got {tuple(activation.shape)} and {tuple(grad.shape)}")
    seq_len = min(int(activation.shape[0]), int(grad.shape[0]))
    score = activation[:seq_len].detach().float().abs() * grad[:seq_len].detach().float().abs()
    return score.mean(dim=-1).cpu()


def group_ranges(groups: Sequence[str | None]) -> list[tuple[int, int, str]]:
    ranges: list[tuple[int, int, str]] = []
    start = None
    current = None
    for idx, group in enumerate(groups):
        if group != current:
            if current is not None and start is not None:
                ranges.append((start, idx - 1, current))
            start = idx
            current = group
    if current is not None and start is not None:
        ranges.append((start, len(groups) - 1, current))
    return ranges


def compact_labels(token_texts: Sequence[str]) -> dict[int, str]:
    labels: dict[int, str] = {}
    sid_begins = [idx for idx, text in enumerate(token_texts) if text == "<|sid_begin|>"]
    sid_ends = [idx for idx, text in enumerate(token_texts) if text == "<|sid_end|>"]
    final_sid_begin = sid_begins[-1] if sid_begins and sid_begins[-1] == len(token_texts) - 1 else None
    history_begins = [idx for idx in sid_begins if idx != final_sid_begin]

    system_start = next((idx for idx, text in enumerate(token_texts[:-1]) if text == "<|im_start|>" and token_texts[idx + 1] == "system"), None)
    user_start = next((idx for idx, text in enumerate(token_texts[:-1]) if text == "<|im_start|>" and token_texts[idx + 1] == "user"), None)
    assistant_start = next((idx for idx, text in enumerate(token_texts[:-1]) if text == "<|im_start|>" and token_texts[idx + 1] == "assistant"), None)
    if system_start is not None:
        labels[system_start] = "system start"
    if user_start is not None:
        labels[user_start] = "user start"
    if history_begins:
        labels[history_begins[0]] = "history SID start"
    if sid_ends:
        labels[sid_ends[-1]] = "history SID end"
        final_instruction = sid_ends[-1] + 1
        while final_instruction < len(token_texts) and token_texts[final_instruction].strip() == "":
            final_instruction += 1
        if final_instruction < len(token_texts):
            labels[final_instruction] = "final instruction"
    if assistant_start is not None:
        labels[assistant_start] = "assistant start"
    if final_sid_begin is not None:
        labels[final_sid_begin] = "predict s_a from <|sid_begin|>"
    return labels


def summarize_groups(values: Sequence[float], groups: Sequence[str | None]) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        if group is not None:
            buckets[group].append(float(value))
    rows: list[dict[str, Any]] = []
    text_mean = None
    if buckets.get("text_prompt"):
        text_mean = sum(buckets["text_prompt"]) / len(buckets["text_prompt"])
    for group, vals in sorted(buckets.items()):
        mean = sum(vals) / len(vals)
        rows.append(
            {
                "token_group": group,
                "num_tokens": len(vals),
                "mean_sensitivity": mean,
                "max_sensitivity": max(vals),
                "ratio_to_text": None if text_mean in (None, 0.0) else mean / text_mean,
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_node(
    *,
    output_dir: Path,
    node: str,
    layers: Sequence[int],
    token_texts: Sequence[str],
    groups: Sequence[str | None],
    labels: Mapping[int, str],
    sensitivities: Mapping[str, Sequence[float]],
    yscale: str,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = list(range(len(token_texts)))
    nrows = len(layers)
    fig, axes = plt.subplots(nrows, 1, figsize=(18, max(2.6 * nrows, 4)), sharex=True, squeeze=False)
    axes_list = list(axes.flat)

    for ax, layer in zip(axes_list, layers):
        key = f"layer{layer}.{node}"
        values = list(sensitivities.get(key, []))
        if not values:
            ax.set_axis_off()
            continue
        positive = [v for v in values if v > 0]
        eps = min(positive) * 0.5 if positive else 1e-12
        y = [max(v, eps) for v in values] if yscale == "log" else values
        ax.plot(x[: len(y)], y, color="#1f77b4", linewidth=1.1, zorder=2)

        for start, end, group in group_ranges(groups):
            color = GROUP_COLORS.get(group)
            if color is None:
                continue
            ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.045, linewidth=0, zorder=0)

        for group, color in GROUP_COLORS.items():
            xs = [idx for idx, item in enumerate(groups[: len(y)]) if item == group]
            if xs:
                ax.scatter(xs, [y[idx] for idx in xs], s=8, color=color, alpha=0.75, label=group, zorder=3)

        for idx, label in labels.items():
            if idx >= len(y):
                continue
            ax.axvline(idx, color="#444444", linewidth=0.45, alpha=0.45)
            ax.text(idx, max(y) if y else 1.0, label, rotation=90, va="top", ha="right", fontsize=6, alpha=0.75)

        if yscale == "log":
            ax.set_yscale("log")
        ax.set_ylabel(f"L{layer}")
        ax.grid(True, linewidth=0.3, alpha=0.35)

    handles, legend_labels = axes_list[0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(legend_labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
    axes_list[-1].set_xlabel("token index")
    fig.suptitle(f"Prefill sensitivity for predicting gt s_a | {node}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.93, 0.96))
    path = output_dir / f"sensitivity__{node}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_from_name(args.dtype),
        trust_remote_code=True,
    ).to(args.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    layers = parse_layer_indices(args.layers, num_layers=len(model.model.layers))
    nodes = parse_str_list(args.nodes)
    capture = ActivationCapture(layers=set(layers), nodes=set(nodes))
    register_hooks(model, capture)
    embedding_handle = install_embedding_grad_anchor(model)

    task_config = get_task_config("ad")
    prompt_token = task_config.get("generation_config", {}).get("prompt_token", "<|sid_begin|>")
    samples = load_ad_samples(
        tokenizer,
        str(resolve_repo_path(args.data_dir)),
        args.split,
        1,
        args.sample_offset,
    )
    sample_id, sample = next(iter(samples.items()))
    target_text = first_sid_target(sample.get("ground_truth", ""), max_sid_tokens=1)
    if target_text is None:
        raise ValueError(f"Sample {sample_id} has no usable SID target")

    prompt = format_prompt(sample["prompt"], prompt_token)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if args.max_prompt_tokens > 0 and prompt_ids.numel() > args.max_prompt_tokens:
        prompt_ids = prompt_ids[-args.max_prompt_tokens :]
    target_ids = tokenizer(target_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if target_ids.numel() != 1:
        raise ValueError(f"Expected the first SID token to encode to one token, got {target_ids.numel()}: {target_text}")

    token_texts = [safe_token_text(tokenizer, token_id) for token_id in prompt_ids.tolist()]
    groups = label_prefill_tokens(token_texts)
    labels = compact_labels(token_texts)

    input_ids = prompt_ids.to(args.device).unsqueeze(0)
    label = target_ids.to(args.device)
    capture.clear()
    model.zero_grad(set_to_none=True)
    try:
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits[0, int(prompt_ids.numel()) - 1, :].float().unsqueeze(0)
        loss = F.cross_entropy(logits, label, reduction="mean")
        loss.backward()

        sensitivities: dict[str, list[float]] = {}
        group_rows: list[dict[str, Any]] = []
        for (layer_idx, node), activation in sorted(capture.activations.items()):
            if activation.grad is None:
                continue
            key = f"layer{layer_idx}.{node}"
            values = compute_token_sensitivity(activation, activation.grad).tolist()
            sensitivities[key] = values
            for row in summarize_groups(values, groups):
                row["key"] = key
                row["layer"] = layer_idx
                row["node"] = node
                group_rows.append(row)
    finally:
        embedding_handle.remove()
        capture.close()
        model.zero_grad(set_to_none=True)

    token_metadata = [
        {
            "index": idx,
            "token": token_text,
            "group": groups[idx],
            "label": labels.get(idx),
        }
        for idx, token_text in enumerate(token_texts)
    ]
    plots = [
        plot_node(
            output_dir=output_dir,
            node=node,
            layers=layers,
            token_texts=token_texts,
            groups=groups,
            labels=labels,
            sensitivities=sensitivities,
            yscale=args.yscale,
        )
        for node in nodes
    ]

    summary = {
        "config": {
            "model_path": args.model_path,
            "data_dir": args.data_dir,
            "split": args.split,
            "sample_offset": args.sample_offset,
            "sample_id": sample_id,
            "dtype": args.dtype,
            "device": args.device,
            "layers": layers,
            "nodes": nodes,
            "max_prompt_tokens": args.max_prompt_tokens,
            "yscale": args.yscale,
            "loss": "CE(logits_at_final_sid_begin, gt_s_a)",
            "target_text": target_text,
            "target_token_id": int(target_ids[0].item()),
        },
        "loss": float(loss.detach().cpu().item()),
        "num_tokens": len(token_texts),
        "plots": plots,
        "group_rows": group_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "sensitivity_values.json").write_text(json.dumps(sensitivities, indent=2), encoding="utf-8")
    (output_dir / "token_metadata.json").write_text(json.dumps(token_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "group_summary.csv", group_rows)

    readme = f"""# Prefill SID Sensitivity Probe\n\nThis directory visualizes per-token prefill sensitivity for one OneRec AD sample.\n\nLoss:\n\n```text\nCE(logits at the final <|sid_begin|> position, ground-truth s_a)\n```\n\nSensitivity:\n\n```text\nmean_channel(|activation| * |dLoss/dactivation|)\n```\n\nSample ID: `{sample_id}`\nTarget: `{target_text}`\nPrompt tokens: `{len(token_texts)}`\n\nThe plots use token index on the x-axis and sensitivity on the y-axis. SID-code tokens are color-marked by group; the final `<|sid_begin|>` is marked as `predict_s_a_position`.\n"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Wrote prefill SID sensitivity plots to: {output_dir}")
    for row in group_rows:
        if row["token_group"] in {"text_prompt", "history_sid_a", "history_sid_b", "history_sid_c", "predict_s_a_position"}:
            ratio = row["ratio_to_text"]
            print(
                f"{row['key']} {row['token_group']} "
                f"mean={row['mean_sensitivity']:.6g} ratio_to_text={ratio}"
            )


if __name__ == "__main__":
    main()
