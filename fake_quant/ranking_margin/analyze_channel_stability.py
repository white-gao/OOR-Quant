#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
for path in (PROJECT_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.tasks.v1_0.registry import get_task_config

try:
    from .channel_stability import (
        canonical_shared_input_name,
        summarize_channel_stability,
        topk_frequency,
        topk_jaccard_matrix,
        topk_overlap_stats,
        topk_selection_mask,
    )
    from .collect_importance import (
        DEFAULT_DATA_DIR,
        DEFAULT_MODEL_PATH,
        dtype_from_name,
        encode_prompt_ids,
        extract_sid_token_ids,
        load_ad_data,
        resolve_input_device,
        token_margin_loss,
    )
except ImportError:
    from fake_quant.ranking_margin.channel_stability import (
        canonical_shared_input_name,
        summarize_channel_stability,
        topk_frequency,
        topk_jaccard_matrix,
        topk_overlap_stats,
        topk_selection_mask,
    )
    from fake_quant.ranking_margin.collect_importance import (
        DEFAULT_DATA_DIR,
        DEFAULT_MODEL_PATH,
        dtype_from_name,
        encode_prompt_ids,
        extract_sid_token_ids,
        load_ad_data,
        resolve_input_device,
        token_margin_loss,
    )

DEFAULT_OUTPUT_DIR = "fake_quant/ranking_margin/channel_stability/ad_sample128_offset1000"


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cross-sample ranking-importance channel stability for selected OneRec nodes."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--sample_size", default=128, type=int)
    parser.add_argument("--sample_offset", default=1000, type=int)
    parser.add_argument("--layers", default="0,7,14,21,27")
    parser.add_argument("--topk_fractions", default="0.01,0.05,0.10")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--negative_rank", type=int, default=32)
    parser.add_argument("--margin_tau", type=float, default=0.0)
    parser.add_argument("--loss", default="softplus", choices=["softplus", "hinge"])
    parser.add_argument("--eta_eps", type=float, default=1e-3)
    parser.add_argument("--max_prompt_tokens", type=int, default=512)
    parser.add_argument("--max_sid_tokens", type=int, default=3)
    parser.add_argument(
        "--save_per_sample",
        action="store_true",
        help="Save per-node per-sample score tensors. This is useful for later plotting but costs more disk.",
    )
    parser.add_argument("--no_plots", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def selected_module_names(layers: list[int]) -> set[str]:
    names: set[str] = set()
    for layer_idx in layers:
        names.add(f"model.layers.{layer_idx}.self_attn.q_proj")
        names.add(f"model.layers.{layer_idx}.mlp.gate_proj")
    return names


def install_selected_hooks(
    model: nn.Module,
    selected_names: set[str],
) -> tuple[dict[str, torch.Tensor], list[Any]]:
    activations: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x) or not x.requires_grad:
                return
            x.retain_grad()
            activations[name] = x

        return hook

    found_names: set[str] = set()
    for name, module in model.named_modules():
        if name not in selected_names or not isinstance(module, nn.Linear):
            continue
        found_names.add(name)
        handles.append(module.register_forward_pre_hook(make_hook(name)))
    missing = sorted(selected_names - found_names)
    if missing:
        raise RuntimeError(f"Selected modules were not found: {missing}")
    return activations, handles


def install_embedding_grad_anchor(model: nn.Module) -> Any:
    embedding = model.get_input_embeddings()

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        return output.detach().requires_grad_(True)

    return embedding.register_forward_hook(hook)


def collect_per_sample_scores(
    model: nn.Module,
    tokenizer: Any,
    test_data: dict[str, dict[str, Any]],
    *,
    input_device: torch.device,
    prompt_token: str,
    layers: list[int],
    negative_rank: int,
    margin_tau: float,
    loss_type: str,
    eta_eps: float,
    max_prompt_tokens: int,
    max_sid_tokens: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    for param in model.parameters():
        param.requires_grad_(False)

    selected_names = selected_module_names(layers)
    activations, handles = install_selected_hooks(model, selected_names)
    embedding_handle = install_embedding_grad_anchor(model)
    per_node_ranking_scores: dict[str, list[torch.Tensor]] = {}
    per_node_activation_scores: dict[str, list[torch.Tensor]] = {}
    skipped = 0

    try:
        iterator = tqdm(test_data.values(), desc="Collect per-sample ranking importance")
        for sample in iterator:
            target_ids = extract_sid_token_ids(tokenizer, sample.get("ground_truth", ""), max_sid_tokens)
            if not target_ids:
                skipped += 1
                continue

            prompt_ids = encode_prompt_ids(
                tokenizer,
                sample["prompt"],
                prompt_token,
                max_prompt_tokens,
                input_device,
            )
            target = torch.tensor(target_ids, device=input_device, dtype=torch.long)
            full_ids = torch.cat([prompt_ids, target], dim=0)
            if full_ids.numel() < 2:
                skipped += 1
                continue

            activations.clear()
            model.zero_grad(set_to_none=True)
            output = model(input_ids=full_ids[:-1].unsqueeze(0), use_cache=False)
            logits = output.logits[0]
            prompt_len = int(prompt_ids.numel())
            loss_terms = []
            for offset, token_id in enumerate(target_ids):
                logit_index = prompt_len - 1 + offset
                if logit_index < 0 or logit_index >= logits.shape[0]:
                    continue
                loss_terms.append(
                    token_margin_loss(
                        logits[logit_index],
                        int(token_id),
                        negative_rank=negative_rank,
                        margin_tau=margin_tau,
                        loss_type=loss_type,
                        eta_eps=eta_eps,
                    )
                )
            if not loss_terms:
                skipped += 1
                continue

            loss = torch.stack(loss_terms).mean()
            loss.backward()

            for module_name, activation in activations.items():
                node_name = canonical_shared_input_name(module_name)
                if node_name is None:
                    continue
                reduce_dims = tuple(range(activation.ndim - 1))
                activation_abs = activation.detach().float().abs()
                activation_score = activation_abs.mean(dim=reduce_dims).cpu()
                per_node_activation_scores.setdefault(node_name, []).append(activation_score)
                grad = activation.grad
                if grad is None:
                    continue
                ranking_score = (activation_abs * grad.detach().float().abs()).mean(dim=reduce_dims).cpu()
                per_node_ranking_scores.setdefault(node_name, []).append(ranking_score)

            del output, logits, loss
            model.zero_grad(set_to_none=True)
    finally:
        embedding_handle.remove()
        for handle in handles:
            handle.remove()

    if not per_node_ranking_scores:
        raise RuntimeError(
            "No per-sample channel importance was collected. "
            f"Skipped samples: {skipped}. Check SID targets and selected layers."
        )
    print(f"Skipped samples without usable SID targets: {skipped}")
    return (
        {name: torch.stack(scores, dim=0) for name, scores in per_node_ranking_scores.items()},
        {name: torch.stack(scores, dim=0) for name, scores in per_node_activation_scores.items()},
    )


def write_summary_csv(path: Path, summaries: dict[str, dict[str, float | int]]) -> None:
    fieldnames = ["node"]
    for summary in summaries.values():
        for key in summary:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for node, summary in sorted(summaries.items()):
            writer.writerow({"node": node, **summary})


def build_overlap_summaries(
    ranking_scores: dict[str, torch.Tensor],
    activation_scores: dict[str, torch.Tensor],
    *,
    topk_fractions: list[float],
) -> dict[str, dict[str, float | int]]:
    summaries: dict[str, dict[str, float | int]] = {}
    for node, scores in sorted(ranking_scores.items()):
        if node not in activation_scores:
            continue
        summary: dict[str, float | int] = {
            "num_samples": int(scores.shape[0]),
            "num_channels": int(scores.shape[1]),
        }
        for fraction in topk_fractions:
            stats = topk_overlap_stats(
                scores,
                activation_scores[node],
                topk_fraction=fraction,
            )
            key = f"topk/{fraction:.6f}"
            for stat_name, value in stats.items():
                summary[f"{key}/{stat_name}"] = value
        summaries[node] = summary
    return summaries


def save_stability_plots(
    output_dir: Path,
    per_node_scores: dict[str, torch.Tensor],
    *,
    topk_fractions: list[float],
    prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for node_name, scores in sorted(per_node_scores.items()):
        safe_name = f"{prefix}_{node_name.replace('.', '_').replace('/', '_')}"
        for topk_fraction in topk_fractions:
            jaccard = topk_jaccard_matrix(scores, topk_fraction).numpy()
            plt.figure(figsize=(6, 5))
            plt.imshow(jaccard, vmin=0.0, vmax=1.0, aspect="auto", cmap="viridis")
            plt.colorbar(label=f"top-{topk_fraction:g} Jaccard")
            plt.xlabel("sample")
            plt.ylabel("sample")
            plt.title(node_name)
            plt.tight_layout()
            plt.savefig(plot_dir / f"{safe_name}_jaccard_top{topk_fraction:g}.png", dpi=160)
            plt.close()

            frequency = topk_frequency(scores, topk_fraction).numpy()
            plt.figure(figsize=(8, 3))
            plt.bar(range(len(frequency)), frequency, width=1.0)
            plt.xlabel("channel")
            plt.ylabel("selected samples")
            plt.title(f"{node_name} top-{topk_fraction:g} frequency")
            plt.tight_layout()
            plt.savefig(plot_dir / f"{safe_name}_frequency_top{topk_fraction:g}.png", dpi=160)
            plt.close()


def save_overlap_plots(
    output_dir: Path,
    ranking_scores: dict[str, torch.Tensor],
    activation_scores: dict[str, torch.Tensor],
    *,
    topk_fractions: list[float],
) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for node_name, scores in sorted(ranking_scores.items()):
        if node_name not in activation_scores:
            continue
        safe_name = node_name.replace(".", "_").replace("/", "_")
        for topk_fraction in topk_fractions:
            ranking_mask = topk_selection_mask(scores, topk_fraction)
            activation_mask = topk_selection_mask(activation_scores[node_name], topk_fraction)
            overlap = (ranking_mask & activation_mask).sum(dim=1).float()
            k = int(ranking_mask.sum(dim=1)[0].item())
            stats = overlap.numpy()
            plt.figure(figsize=(7, 3))
            plt.bar(range(len(stats)), stats, width=1.0)
            plt.axhline(k, color="black", linestyle="--", linewidth=1, label="max overlap")
            plt.xlabel("sample")
            plt.ylabel("overlap channels")
            plt.title(f"{node_name} ranking vs activation top-{topk_fraction:g}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"overlap_{safe_name}_top{topk_fraction:g}.png", dpi=160)
            plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    layers = parse_int_list(args.layers)
    topk_fractions = parse_float_list(args.topk_fractions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": dtype_from_name(args.dtype),
        "trust_remote_code": True,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    if not args.device_map:
        model = model.to(args.device)
    model.eval()

    input_device = resolve_input_device(model, args.device)
    task_config = get_task_config("ad")
    prompt_token = task_config.get("generation_config", {}).get("prompt_token", "<|sid_begin|>")
    test_data = load_ad_data(
        tokenizer,
        args.data_dir,
        args.split,
        args.sample_size,
        sample_offset=args.sample_offset,
    )

    per_node_ranking_scores, per_node_activation_scores = collect_per_sample_scores(
        model,
        tokenizer,
        test_data,
        input_device=input_device,
        prompt_token=prompt_token,
        layers=layers,
        negative_rank=args.negative_rank,
        margin_tau=args.margin_tau,
        loss_type=args.loss,
        eta_eps=args.eta_eps,
        max_prompt_tokens=args.max_prompt_tokens,
        max_sid_tokens=args.max_sid_tokens,
    )
    ranking_summaries = {
        node: summarize_channel_stability(scores, topk_fractions=topk_fractions)
        for node, scores in sorted(per_node_ranking_scores.items())
    }
    activation_summaries = {
        node: summarize_channel_stability(scores, topk_fractions=topk_fractions)
        for node, scores in sorted(per_node_activation_scores.items())
    }
    overlap_summaries = build_overlap_summaries(
        per_node_ranking_scores,
        per_node_activation_scores,
        topk_fractions=topk_fractions,
    )
    metadata = {
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "split": args.split,
        "sample_size": args.sample_size,
        "sample_offset": args.sample_offset,
        "sample_range": [args.sample_offset, args.sample_offset + args.sample_size],
        "layers": layers,
        "nodes": ["attn_qkv_input", "ffn_gate_up_input"],
        "topk_fractions": topk_fractions,
        "dtype": args.dtype,
        "device": args.device,
        "seed": args.seed,
        "negative_rank": args.negative_rank,
        "margin_tau": args.margin_tau,
        "loss": args.loss,
        "eta_eps": args.eta_eps,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_sid_tokens": args.max_sid_tokens,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "ranking_importance": ranking_summaries,
                "activation_outlier": activation_summaries,
                "ranking_activation_overlap": overlap_summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary_csv(output_dir / "ranking_summary.csv", ranking_summaries)
    write_summary_csv(output_dir / "activation_summary.csv", activation_summaries)
    write_summary_csv(output_dir / "overlap_summary.csv", overlap_summaries)
    write_summary_csv(output_dir / "summary.csv", ranking_summaries)
    if args.save_per_sample:
        torch.save(
            {
                "metadata": metadata,
                "ranking_scores": per_node_ranking_scores,
                "activation_scores": per_node_activation_scores,
            },
            output_dir / "per_sample_importance.pt",
        )
    if not args.no_plots:
        save_stability_plots(
            output_dir,
            per_node_ranking_scores,
            topk_fractions=topk_fractions,
            prefix="ranking",
        )
        save_stability_plots(
            output_dir,
            per_node_activation_scores,
            topk_fractions=topk_fractions,
            prefix="activation",
        )
        save_overlap_plots(
            output_dir,
            per_node_ranking_scores,
            per_node_activation_scores,
            topk_fractions=topk_fractions,
        )

    print(f"Saved channel stability analysis to: {output_dir}")
    print(f"Collected nodes: {len(per_node_ranking_scores)}")


if __name__ == "__main__":
    main()
