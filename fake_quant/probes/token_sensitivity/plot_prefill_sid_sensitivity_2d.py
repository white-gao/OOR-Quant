#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.tasks.v1_0.registry import get_task_config
from fake_quant.probes.token_sensitivity.plot_prefill_sid_sensitivity import (
    GROUP_COLORS,
    compact_labels,
    group_ranges,
    label_prefill_tokens,
)
from fake_quant.probes.token_sensitivity.probe_sid_token_sensitivity import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_PATH,
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


DEFAULT_OUTPUT_DIR = "fake_quant_learnable/results/analysis/token_sensitivity/prefill_sample0_s_a_loss_token_channel_full"
DEFAULT_LAYERS = "0,8,16,24,27"
DEFAULT_NODES = "attn_qkv_input,ffn_gate_up_input,block_output"
PROFILE_GROUPS = (
    "all_tokens",
    "text_prompt",
    "history_sid_a",
    "history_sid_b",
    "history_sid_c",
    "predict_s_a_position",
)
PROFILE_COLORS = {
    "all_tokens": "#111111",
    "text_prompt": GROUP_COLORS["text_prompt"],
    "history_sid_a": GROUP_COLORS["history_sid_a"],
    "history_sid_b": GROUP_COLORS["history_sid_b"],
    "history_sid_c": GROUP_COLORS["history_sid_c"],
    "predict_s_a_position": GROUP_COLORS["predict_s_a_position"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot full token-channel prefill sensitivity heatmaps for predicting OneRec gt s_a."
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
    parser.add_argument("--save_matrices", action="store_true", help="Save full sensitivity matrices as a .pt file.")
    parser.add_argument("--profile_exclude_groups", default="", help="Comma-separated token groups hidden only in channel profile plots.")
    parser.add_argument("--profile_plot_suffix", default="", help="Optional suffix for channel profile plot filenames.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_token_channel_sensitivity(activation: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Return the full [token, channel] sensitivity matrix |activation| * |gradient|."""
    if activation.ndim >= 3:
        activation = activation[0]
    if grad.ndim >= 3:
        grad = grad[0]
    if activation.ndim != 2 or grad.ndim != 2:
        raise ValueError(f"Expected [seq, hidden] tensors, got {tuple(activation.shape)} and {tuple(grad.shape)}")
    seq_len = min(int(activation.shape[0]), int(grad.shape[0]))
    hidden = min(int(activation.shape[1]), int(grad.shape[1]))
    return (
        activation[:seq_len, :hidden].detach().float().abs()
        * grad[:seq_len, :hidden].detach().float().abs()
    ).cpu()


def compute_channel_profiles(matrix: torch.Tensor, groups: Sequence[str | None]) -> dict[str, torch.Tensor]:
    """Average a full [token, channel] matrix over token groups, preserving all channels."""
    if matrix.ndim != 2:
        raise ValueError(f"Expected [token, channel] matrix, got {tuple(matrix.shape)}")
    seq_len = min(int(matrix.shape[0]), len(groups))
    matrix = matrix[:seq_len].float().cpu()
    profiles: dict[str, torch.Tensor] = {}
    profiles["all_tokens"] = matrix.mean(dim=0)
    for group in sorted({group for group in groups[:seq_len] if group is not None}):
        indices = [idx for idx, item in enumerate(groups[:seq_len]) if item == group]
        if indices:
            profiles[group] = matrix[indices].mean(dim=0)
    return profiles


def profile_summary_rows(key: str, profiles: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, values in profiles.items():
        values = values.float()
        rows.append(
            {
                "key": key,
                "token_group": group,
                "num_channels": int(values.numel()),
                "mean_channel_sensitivity": float(values.mean().item()),
                "max_channel_sensitivity": float(values.max().item()),
                "p99_channel_sensitivity": float(torch.quantile(values, 0.99).item()),
                "top_channel": int(torch.argmax(values).item()),
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


def heatmap_values(matrix: torch.Tensor) -> tuple[Any, float]:
    import numpy as np

    arr = matrix.float().numpy()
    positive = arr[arr > 0]
    eps = float(positive.min() * 0.5) if positive.size else 1e-30
    return np.log10(arr + eps).T, eps


def add_token_annotations(ax: Any, groups: Sequence[str | None], labels: Mapping[int, str], *, y_top: int) -> None:
    for start, end, group in group_ranges(groups):
        color = GROUP_COLORS.get(group)
        if color is not None:
            ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.06, linewidth=0)
    for idx, label in labels.items():
        ax.axvline(idx, color="white", linewidth=0.45, alpha=0.7)
        ax.text(idx, y_top, label, rotation=90, va="top", ha="right", fontsize=5.5, color="white", alpha=0.85)


def plot_heatmaps(
    *,
    output_dir: Path,
    node: str,
    layers: Sequence[int],
    matrices: Mapping[str, torch.Tensor],
    groups: Sequence[str | None],
    labels: Mapping[int, str],
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows = len(layers)
    fig, axes = plt.subplots(nrows, 1, figsize=(18, max(3.2 * nrows, 5)), sharex=True, squeeze=False)
    axes_list = list(axes.flat)
    image = None
    eps_used = None
    for ax, layer in zip(axes_list, layers):
        key = f"layer{layer}.{node}"
        matrix = matrices.get(key)
        if matrix is None:
            ax.set_axis_off()
            continue
        values, eps = heatmap_values(matrix)
        eps_used = eps
        image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap="magma", origin="lower")
        add_token_annotations(ax, groups, labels, y_top=values.shape[0] - 1)
        ax.set_ylabel(f"L{layer}\nchannel")
        ax.set_title(f"{key} full token-channel sensitivity", fontsize=9)
    axes_list[-1].set_xlabel("token index")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes_list, fraction=0.015, pad=0.01)
        cbar.set_label(f"log10(|activation|*|gradient| + eps), eps={eps_used:.2e}")
    fig.suptitle(f"Full token-channel prefill sensitivity | {node}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.965, 0.965))
    path = output_dir / f"token_channel_heatmap__{node}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def plot_channel_profiles(
    *,
    output_dir: Path,
    node: str,
    layers: Sequence[int],
    profile_map: Mapping[str, Mapping[str, torch.Tensor]],
    profile_groups: Sequence[str] = PROFILE_GROUPS,
    filename_suffix: str = "",
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows = len(layers)
    fig, axes = plt.subplots(nrows, 1, figsize=(18, max(2.6 * nrows, 4)), sharex=True, squeeze=False)
    axes_list = list(axes.flat)
    for ax, layer in zip(axes_list, layers):
        key = f"layer{layer}.{node}"
        profiles = profile_map.get(key)
        if not profiles:
            ax.set_axis_off()
            continue
        for group in profile_groups:
            values = profiles.get(group)
            if values is None:
                continue
            positive = values[values > 0]
            eps = float(positive.min().item() * 0.5) if positive.numel() else 1e-30
            y = torch.log10(values.float() + eps).numpy()
            ax.plot(y, linewidth=0.8 if group != "all_tokens" else 1.2, alpha=0.85, color=PROFILE_COLORS.get(group), label=group)
        ax.set_ylabel(f"L{layer}")
        ax.grid(True, linewidth=0.3, alpha=0.35)
    axes_list[-1].set_xlabel("channel index")
    handles, legend_labels = axes_list[0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(legend_labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
    fig.suptitle(f"Channel sensitivity profiles | {node}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.93, 0.96))
    suffix = f"_{filename_suffix}" if filename_suffix else ""
    path = output_dir / f"channel_profile{suffix}__{node}.png"
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
        raise ValueError(f"Expected first SID token to be one token, got {target_ids.numel()}: {target_text}")

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

        matrices: dict[str, torch.Tensor] = {}
        profiles: dict[str, dict[str, torch.Tensor]] = {}
        profile_rows: list[dict[str, Any]] = []
        for (layer_idx, node), activation in sorted(capture.activations.items()):
            if activation.grad is None:
                continue
            key = f"layer{layer_idx}.{node}"
            matrix = compute_token_channel_sensitivity(activation, activation.grad)
            matrices[key] = matrix
            profiles[key] = compute_channel_profiles(matrix, groups)
            for row in profile_summary_rows(key, profiles[key]):
                row["layer"] = layer_idx
                row["node"] = node
                profile_rows.append(row)
    finally:
        embedding_handle.remove()
        capture.close()
        model.zero_grad(set_to_none=True)

    token_metadata = [
        {"index": idx, "token": token_text, "group": groups[idx], "label": labels.get(idx)}
        for idx, token_text in enumerate(token_texts)
    ]
    heatmap_plots = [
        plot_heatmaps(output_dir=output_dir, node=node, layers=layers, matrices=matrices, groups=groups, labels=labels)
        for node in nodes
    ]
    excluded_profile_groups = set(parse_str_list(args.profile_exclude_groups))
    profile_groups = [group for group in PROFILE_GROUPS if group not in excluded_profile_groups]
    profile_plots = [
        plot_channel_profiles(
            output_dir=output_dir,
            node=node,
            layers=layers,
            profile_map=profiles,
            profile_groups=profile_groups,
            filename_suffix=args.profile_plot_suffix,
        )
        for node in nodes
    ]

    if args.save_matrices:
        torch.save({key: value.half() for key, value in matrices.items()}, output_dir / "sensitivity_matrices_fp16.pt")

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
            "profile_exclude_groups": sorted(excluded_profile_groups),
            "profile_plot_suffix": args.profile_plot_suffix,
            "loss": "CE(logits_at_final_sid_begin, gt_s_a)",
            "target_text": target_text,
            "target_token_id": int(target_ids[0].item()),
            "matrix_definition": "full [token, channel] |activation| * |gradient|; no token/channel filtering",
        },
        "loss": float(loss.detach().cpu().item()),
        "num_tokens": len(token_texts),
        "num_channels_by_key": {key: int(value.shape[1]) for key, value in matrices.items()},
        "heatmap_plots": heatmap_plots,
        "channel_profile_plots": profile_plots,
        "profile_rows": profile_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "token_metadata.json").write_text(json.dumps(token_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "channel_profile_summary.csv", profile_rows)

    readme = f"""# Full Token-Channel Prefill SID Sensitivity\n\nThis directory contains full token-channel sensitivity visualizations for one OneRec AD sample.\n\nLoss:\n\n```text\nCE(logits at the final <|sid_begin|> position, ground-truth s_a)\n```\n\nMatrix definition:\n\n```text\nS[token, channel] = |activation[token, channel]| * |dLoss/dactivation[token, channel]|\n```\n\nNo token or channel filtering is applied in the heatmaps. The heatmap color is `log10(S + eps)` only for readability.\n\nSample ID: `{sample_id}`\nTarget: `{target_text}`\nPrompt tokens: `{len(token_texts)}`\nLayers: `{layers}`\nNodes: `{nodes}`\n\nFiles:\n\n```text\ntoken_channel_heatmap__*.png   full token x channel heatmaps\nchannel_profile__*.png         channel-wise profiles averaged by token group\nchannel_profile_summary.csv    numeric profile summary\ntoken_metadata.json            token index/group/label metadata\nsummary.json                   run config and plot paths\n```\n"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Wrote full token-channel sensitivity outputs to: {output_dir}")
    for path in heatmap_plots + profile_plots:
        print(path)


if __name__ == "__main__":
    main()
