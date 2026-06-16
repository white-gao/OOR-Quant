#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
for path in (PROJECT_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.tasks.v1_0.registry import get_loader, get_task_config  # noqa: E402


DEFAULT_MODEL_PATH = "/home/guowei/OneRec-1.7B/"
DEFAULT_DATA_DIR = "data/onerec_data/benchmark-data-calib1024"
DEFAULT_OUTPUT_DIR = "fake_quant_learnable/results/analysis/token_sensitivity/sid_tf_ce_sample16_layer_last"
DEFAULT_NODES = "block_output"
SID_ITEM_RE = re.compile(r"<\|sid_begin\|>(?P<sid>(?:<s_[abc]_[^>]+>)+)<\|sid_end\|>")
SID_CODE_RE = re.compile(r"^<s_([abc])_[^>]+>$")
SID_BOUNDARY_TOKENS = {"<|sid_begin|>", "<|sid_end|>"}
CHAT_SPECIAL_TOKENS = {"<|im_start|>", "<|im_end|>", "<think>", "</think>"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe OneRec token-group sensitivity with teacher-forced SID CE gradients."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="calib", choices=["calib", "test"])
    parser.add_argument("--sample_size", type=int, default=16)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="last", help='Comma-separated layers, "last", or "all".')
    parser.add_argument("--nodes", default=DEFAULT_NODES)
    parser.add_argument("--max_prompt_tokens", type=int, default=0, help="Left-truncate prompt tokens; 0 disables truncation.")
    parser.add_argument("--max_sid_tokens", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_layer_indices(value: str, *, num_layers: int) -> list[int]:
    value = value.strip().lower()
    if value == "all":
        return list(range(num_layers))
    if value == "last":
        return [num_layers - 1]
    indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    for index in indices:
        if index < 0 or index >= num_layers:
            raise ValueError(f"Layer index {index} out of range [0, {num_layers})")
    return indices


def format_prompt(prompt: str, prompt_token: str) -> str:
    if prompt_token and not prompt.endswith(prompt_token):
        return prompt + prompt_token
    return prompt


def load_ad_samples(
    tokenizer: Any,
    data_dir: str,
    split: str,
    sample_size: int,
    sample_offset: int,
) -> dict[str, Mapping[str, Any]]:
    if sample_size <= 0:
        raise ValueError("--sample_size must be positive")
    if sample_offset < 0:
        raise ValueError("--sample_offset must be non-negative")
    loader = get_loader("ad", data_dir=data_dir, tokenizer=tokenizer, enable_thinking=False)
    data = loader.load_data(split=split, sample_size=sample_size + sample_offset)
    return dict(list(data.items())[sample_offset : sample_offset + sample_size])


def first_sid_target(ground_truth: str, *, max_sid_tokens: int) -> str | None:
    match = SID_ITEM_RE.search(ground_truth or "")
    if not match:
        return None
    tokens = re.findall(r"<s_[abc]_[^>]+>", match.group("sid"))
    if not tokens:
        return None
    return "".join(tokens[:max_sid_tokens])


def safe_token_text(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return text.replace("\n", "\\n").replace("\r", "\\r")


def label_teacher_forced_tokens(
    token_texts: Sequence[str],
    *,
    prompt_len: int,
    target_len: int,
) -> list[str | None]:
    groups: list[str | None] = []
    for idx, token_text in enumerate(token_texts):
        if idx == prompt_len - 1:
            groups.append("predict_s_a_position")
            continue
        if prompt_len <= idx < prompt_len + max(target_len - 1, 0):
            offset = idx - prompt_len
            if offset == 0:
                groups.append("predict_s_b_position")
            elif offset == 1:
                groups.append("predict_s_c_position")
            else:
                groups.append("predict_extra_position")
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


@dataclass
class SensitivityStats:
    num_tokens: int = 0
    num_values: int = 0
    abs_grad_sum: float = 0.0
    abs_act_sum: float = 0.0
    abs_act_grad_sum: float = 0.0


class SensitivityAccumulator:
    def __init__(self) -> None:
        self._stats: dict[tuple[str, str], SensitivityStats] = defaultdict(SensitivityStats)

    def add(
        self,
        key: str,
        activation: torch.Tensor,
        grad: torch.Tensor,
        groups: Sequence[str | None],
    ) -> None:
        if activation.ndim >= 3:
            activation = activation[0]
        if grad.ndim >= 3:
            grad = grad[0]
        if activation.ndim != 2 or grad.ndim != 2:
            return
        seq_len = min(int(activation.shape[0]), int(grad.shape[0]), len(groups))
        activation = activation[:seq_len].detach().float().cpu()
        grad = grad[:seq_len].detach().float().cpu()

        positions_by_group: dict[str, list[int]] = defaultdict(list)
        for idx, group in enumerate(groups[:seq_len]):
            if group is not None:
                positions_by_group[group].append(idx)

        for group, positions in positions_by_group.items():
            act_slice = activation[positions]
            grad_slice = grad[positions]
            stat = self._stats[(key, group)]
            stat.num_tokens += len(positions)
            stat.num_values += int(act_slice.numel())
            stat.abs_grad_sum += float(grad_slice.abs().sum().item())
            stat.abs_act_sum += float(act_slice.abs().sum().item())
            stat.abs_act_grad_sum += float((act_slice.abs() * grad_slice.abs()).sum().item())

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (key, group), stat in sorted(self._stats.items()):
            denom = max(stat.num_values, 1)
            rows.append(
                {
                    "key": key,
                    "token_group": group,
                    "num_tokens": stat.num_tokens,
                    "num_values": stat.num_values,
                    "mean_abs_grad": stat.abs_grad_sum / denom,
                    "mean_abs_activation": stat.abs_act_sum / denom,
                    "mean_abs_act_grad": stat.abs_act_grad_sum / denom,
                }
            )

        text_baseline = {
            row["key"]: row
            for row in rows
            if row["token_group"] == "text_prompt"
        }
        for row in rows:
            baseline = text_baseline.get(row["key"])
            if baseline is None:
                row["grad_ratio_to_text"] = None
                row["act_grad_ratio_to_text"] = None
                continue
            row["grad_ratio_to_text"] = ratio_or_none(
                row["mean_abs_grad"], baseline["mean_abs_grad"]
            )
            row["act_grad_ratio_to_text"] = ratio_or_none(
                row["mean_abs_act_grad"], baseline["mean_abs_act_grad"]
            )
        return rows


def ratio_or_none(value: float, baseline: float) -> float | None:
    if not math.isfinite(value) or not math.isfinite(baseline) or baseline == 0:
        return None
    return value / baseline


class ActivationCapture:
    def __init__(self, *, layers: set[int], nodes: set[str]) -> None:
        self.layers = layers
        self.nodes = nodes
        self.handles: list[Any] = []
        self.activations: dict[tuple[int, str], torch.Tensor] = {}

    def add_output_hook(self, module: nn.Module, layer: int, node: str) -> None:
        if layer not in self.layers or node not in self.nodes:
            return

        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            self._store(layer, node, tensor)

        self.handles.append(module.register_forward_hook(hook))

    def add_input_hook(self, module: nn.Module, layer: int, node: str) -> None:
        if layer not in self.layers or node not in self.nodes:
            return

        def hook(_module: nn.Module, inputs: Any) -> None:
            if inputs:
                self._store(layer, node, inputs[0])

        self.handles.append(module.register_forward_pre_hook(hook))

    def _store(self, layer: int, node: str, tensor: Any) -> None:
        if not torch.is_tensor(tensor) or not tensor.requires_grad:
            return
        tensor.retain_grad()
        self.activations[(layer, node)] = tensor

    def clear(self) -> None:
        self.activations.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def register_hooks(model: nn.Module, capture: ActivationCapture) -> None:
    for layer_idx, layer in enumerate(model.model.layers):
        capture.add_output_hook(layer.input_layernorm, layer_idx, "attn_qkv_input")
        capture.add_input_hook(layer.self_attn.o_proj, layer_idx, "attn_o_input")
        capture.add_output_hook(layer.post_attention_layernorm, layer_idx, "ffn_gate_up_input")
        capture.add_input_hook(layer.mlp.down_proj, layer_idx, "ffn_down_input")
        capture.add_output_hook(layer, layer_idx, "block_output")


def install_embedding_grad_anchor(model: nn.Module) -> Any:
    embedding = model.get_input_embeddings()

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        return output.detach().requires_grad_(True)

    return embedding.register_forward_hook(hook)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    accumulator = SensitivityAccumulator()

    task_config = get_task_config("ad")
    prompt_token = task_config.get("generation_config", {}).get("prompt_token", "<|sid_begin|>")
    samples = load_ad_samples(
        tokenizer,
        str(resolve_repo_path(args.data_dir)),
        args.split,
        args.sample_size,
        args.sample_offset,
    )

    skipped = 0
    losses: list[float] = []
    try:
        for sample in tqdm(samples.values(), desc="Probe SID token sensitivity"):
            target_text = first_sid_target(sample.get("ground_truth", ""), max_sid_tokens=args.max_sid_tokens)
            if target_text is None:
                skipped += 1
                continue

            prompt = format_prompt(sample["prompt"], prompt_token)
            prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            if args.max_prompt_tokens > 0 and prompt_ids.numel() > args.max_prompt_tokens:
                prompt_ids = prompt_ids[-args.max_prompt_tokens :]
            target_ids = tokenizer(target_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            if target_ids.numel() == 0:
                skipped += 1
                continue

            input_ids = torch.cat([prompt_ids, target_ids[:-1]], dim=0).to(args.device)
            labels = target_ids.to(args.device)
            prompt_len = int(prompt_ids.numel())
            target_len = int(target_ids.numel())
            positions = torch.arange(
                prompt_len - 1,
                prompt_len - 1 + target_len,
                device=args.device,
            )
            token_texts = [safe_token_text(tokenizer, token_id) for token_id in input_ids.detach().cpu().tolist()]
            token_groups = label_teacher_forced_tokens(
                token_texts,
                prompt_len=prompt_len,
                target_len=target_len,
            )

            capture.clear()
            model.zero_grad(set_to_none=True)
            outputs = model(input_ids=input_ids.unsqueeze(0), use_cache=False)
            logits = outputs.logits[0, positions, :].float()
            loss = F.cross_entropy(logits, labels, reduction="mean")
            loss.backward()
            losses.append(float(loss.detach().cpu().item()))

            for (layer_idx, node), activation in capture.activations.items():
                if activation.grad is None:
                    continue
                accumulator.add(
                    f"layer{layer_idx}.{node}",
                    activation,
                    activation.grad,
                    token_groups,
                )

            del outputs, logits, loss
            model.zero_grad(set_to_none=True)
    finally:
        embedding_handle.remove()
        capture.close()

    rows = accumulator.rows()
    payload = {
        "config": {
            "model_path": args.model_path,
            "data_dir": args.data_dir,
            "split": args.split,
            "sample_size": args.sample_size,
            "sample_offset": args.sample_offset,
            "dtype": args.dtype,
            "device": args.device,
            "layers": layers,
            "nodes": nodes,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_sid_tokens": args.max_sid_tokens,
            "seed": args.seed,
        },
        "skipped_samples": skipped,
        "mean_sid_ce_loss": sum(losses) / len(losses) if losses else None,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "summary.csv", rows)
    print(f"Wrote token sensitivity summary to: {output_dir}")
    for row in rows:
        print(
            f"{row['key']} {row['token_group']} "
            f"grad={row['mean_abs_grad']:.6g} grad/text={row['grad_ratio_to_text']} "
            f"act_grad={row['mean_abs_act_grad']:.6g} act_grad/text={row['act_grad_ratio_to_text']}"
        )


if __name__ == "__main__":
    main()
