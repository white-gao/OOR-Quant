#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
for path in (PROJECT_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.tasks.v1_0.registry import get_task_config

from fake_quant.probes.activation_probe.profile_ad_activations import (
    encode_prompt,
    load_ad_data,
    parse_sample_size,
    resolve_data_dir,
    resolve_model_path,
    torch_dtype,
    write_json,
)
from fake_quant.probes.activation_probe.profile_teacher_forced_decode_steps import (
    first_ground_truth_sid,
    token_id_for_text,
)


DEFAULT_MODEL_PATH = "/home/guowei/OneRec-1.7B/"
DEFAULT_DATA_DIR = "data/onerec_data/benchmark-data"
DEFAULT_OUTPUT_DIR = (
    "fake_quant/probes/activation_probe/activation_profiles/v1.0/"
    "OneRec-1.7B-ad-teacher-forced-channel-stability-sample-128"
)
DEFAULT_NODES = [
    "attn_qkv_input",
    "attn_o_input",
    "ffn_gate_up_input",
    "ffn_down_input",
    "residual_block_output",
    "final_norm",
]
STAGES = ["predict_a", "predict_b", "predict_c", "predict_end"]


@dataclass
class CaptureContext:
    sample_id: str
    stage: str


class TeacherForcedChannelCapture:
    def __init__(self, *, layers: set[int], nodes: set[str], final_layer_idx: int) -> None:
        self.layers = layers
        self.nodes = nodes
        self.final_layer_idx = final_layer_idx
        self.context: CaptureContext | None = None
        self.handles: List[Any] = []
        self.scores: Dict[tuple[str, int, str], List[tuple[str, torch.Tensor]]] = {}

    def add_output_hook(self, module: torch.nn.Module, layer: int, node: str) -> None:
        if not self._enabled(layer, node):
            return

        def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            self._store(layer, node, tensor)

        self.handles.append(module.register_forward_hook(hook))

    def add_input_hook(self, module: torch.nn.Module, layer: int, node: str) -> None:
        if not self._enabled(layer, node):
            return

        def hook(_module: torch.nn.Module, inputs: Any, _output: Any) -> None:
            if inputs:
                self._store(layer, node, inputs[0])

        self.handles.append(module.register_forward_hook(hook))

    def _enabled(self, layer: int, node: str) -> bool:
        return layer in self.layers and node in self.nodes

    def _store(self, layer: int, node: str, tensor: Any) -> None:
        context = self.context
        if context is None or not torch.is_tensor(tensor):
            return
        if tensor.dim() >= 3:
            tensor = tensor[0]
        if tensor.dim() == 2:
            tensor = tensor[-1]
        if tensor.dim() != 1:
            return
        values = tensor.detach().float().abs().cpu()
        self.scores.setdefault((context.stage, layer, node), []).append((context.sample_id, values))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cross-sample channel stability for teacher-forced SID decode positions."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample_size", default="128")
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max_tokens", type=int, default=0)
    parser.add_argument("--layers", default="0,7,14,21,27,28")
    parser.add_argument("--nodes", default=",".join(DEFAULT_NODES))
    parser.add_argument("--topk_counts", default="32")
    parser.add_argument("--topk_fractions", default="0.01,0.05")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_int_list(value: str, *, max_layer: int | None = None) -> List[int]:
    if value == "all":
        if max_layer is None:
            raise ValueError("max_layer is required when parsing 'all'")
        return list(range(max_layer + 1))
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def register_capture_hooks(model: torch.nn.Module, capture: TeacherForcedChannelCapture) -> None:
    for layer_idx, layer in enumerate(model.model.layers):
        capture.add_output_hook(layer.input_layernorm, layer_idx, "attn_qkv_input")
        capture.add_input_hook(layer.self_attn.o_proj, layer_idx, "attn_o_input")
        capture.add_output_hook(layer.post_attention_layernorm, layer_idx, "ffn_gate_up_input")
        capture.add_input_hook(layer.mlp.down_proj, layer_idx, "ffn_down_input")
        capture.add_output_hook(layer, layer_idx, "residual_block_output")
    capture.add_output_hook(model.model.norm, capture.final_layer_idx, "final_norm")


def topk_count_from_fraction(num_channels: int, fraction: float) -> int:
    return min(num_channels, max(1, int(math.ceil(num_channels * fraction))))


def topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, scores.shape[-1])
    indices = torch.topk(scores.float(), k=k, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter_(-1, indices, True)


def offdiag_mean(matrix: torch.Tensor) -> float:
    if matrix.shape[0] <= 1:
        return 1.0
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return float(matrix[mask].mean().item())


def summarize_scores(scores: torch.Tensor, *, topk_specs: Sequence[tuple[str, int]]) -> Dict[str, Any]:
    scores = scores.float()
    mean = scores.mean(dim=0)
    std = scores.std(dim=0, unbiased=False)
    cv = std / mean.abs().clamp_min(1e-12)
    row: Dict[str, Any] = {
        "num_samples": int(scores.shape[0]),
        "num_channels": int(scores.shape[1]),
        "score_mean": float(scores.mean().item()),
        "score_max": float(scores.max().item()),
        "channel_mean_max": float(mean.max().item()),
        "cv_mean": float(cv.mean().item()),
        "cv_median": float(cv.median().item()),
    }
    for label, k in topk_specs:
        mask = topk_mask(scores, k)
        intersection = (mask[:, None, :] & mask[None, :, :]).sum(dim=-1).float()
        union = (mask[:, None, :] | mask[None, :, :]).sum(dim=-1).float().clamp_min(1.0)
        jaccard = intersection / union
        frequency = mask.float().sum(dim=0)
        random_jaccard = k / max(1, (2 * scores.shape[1] - k))
        row[f"{label}_k"] = int(k)
        row[f"{label}_jaccard_mean"] = offdiag_mean(jaccard)
        row[f"{label}_random_jaccard"] = float(random_jaccard)
        row[f"{label}_frequency_max"] = float(frequency.max().item())
        row[f"{label}_frequency_max_ratio"] = float(frequency.max().item() / scores.shape[0])
    return row


def jaccard_between_vectors(lhs: torch.Tensor, rhs: torch.Tensor, k: int) -> float:
    lhs_mask = topk_mask(lhs.float().unsqueeze(0), k)[0]
    rhs_mask = topk_mask(rhs.float().unsqueeze(0), k)[0]
    intersection = (lhs_mask & rhs_mask).sum().float()
    union = (lhs_mask | rhs_mask).sum().float().clamp_min(1.0)
    return float((intersection / union).item())


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def profile_sample(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    capture: TeacherForcedChannelCapture,
    sample_id: str,
    prompt: str,
    ground_truth: str,
    prompt_token: str,
    device: str,
    max_tokens: int,
) -> None:
    target_texts = first_ground_truth_sid(ground_truth)
    target_ids = [token_id_for_text(tokenizer, token_text) for token_text in target_texts]
    input_ids = encode_prompt(tokenizer, prompt, prompt_token, max_tokens).to(device)

    with torch.inference_mode():
        capture.context = CaptureContext(sample_id=sample_id, stage="predict_a")
        output = model(input_ids=input_ids, use_cache=True)
        past_key_values = output.past_key_values

        for index, stage in enumerate(STAGES[1:], start=1):
            input_token = torch.tensor([[target_ids[index - 1]]], device=device)
            capture.context = CaptureContext(sample_id=sample_id, stage=stage)
            output = model(input_ids=input_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
    capture.context = None


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    model_path = resolve_model_path(args.model_path)
    data_dir = resolve_data_dir(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype(args.dtype),
        device_map=None,
    ).to(args.device)
    model.eval()

    final_layer_idx = len(model.model.layers)
    layers = set(parse_int_list(args.layers, max_layer=final_layer_idx))
    nodes = set(parse_str_list(args.nodes))
    topk_counts = parse_int_list(args.topk_counts)
    topk_fractions = parse_float_list(args.topk_fractions)

    sample_size = parse_sample_size(args.sample_size)
    test_data = load_ad_data(tokenizer, data_dir, args.split, sample_size)
    prompt_token = get_task_config("ad").get("generation_config", {}).get("prompt_token", "<|sid_begin|>")

    capture = TeacherForcedChannelCapture(layers=layers, nodes=nodes, final_layer_idx=final_layer_idx)
    register_capture_hooks(model, capture)
    skipped_samples: List[Dict[str, str]] = []
    start = time.time()
    try:
        for sample_id, sample in tqdm(test_data.items(), desc="Teacher-forced channel stability"):
            try:
                profile_sample(
                    model=model,
                    tokenizer=tokenizer,
                    capture=capture,
                    sample_id=sample_id,
                    prompt=sample["prompt"],
                    ground_truth=sample.get("ground_truth", ""),
                    prompt_token=prompt_token,
                    device=args.device,
                    max_tokens=args.max_tokens,
                )
            except ValueError as exc:
                skipped_samples.append({"sample_id": sample_id, "reason": str(exc)})
    finally:
        capture.close()
    elapsed = time.time() - start

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensor_payload: Dict[str, Any] = {
        "format": "oor_teacher_forced_channel_stability_v1",
        "config": vars(args),
        "sample_ids": {},
        "scores": {},
    }
    stability_rows: List[Dict[str, Any]] = []
    mean_scores: Dict[tuple[str, int, str], torch.Tensor] = {}

    for (stage, layer, node), items in sorted(capture.scores.items()):
        sample_ids = [sample_id for sample_id, _ in items]
        scores = torch.stack([score for _, score in items], dim=0)
        key = f"{stage}/layer_{layer}/{node}"
        tensor_payload["sample_ids"][key] = sample_ids
        tensor_payload["scores"][key] = scores
        mean_scores[(stage, layer, node)] = scores.mean(dim=0)

        specs = [(f"top{count}", min(count, scores.shape[1])) for count in topk_counts]
        specs.extend(
            (f"top{fraction:g}", topk_count_from_fraction(scores.shape[1], fraction))
            for fraction in topk_fractions
        )
        row = {"stage": stage, "layer": layer, "node": node}
        row.update(summarize_scores(scores, topk_specs=specs))
        stability_rows.append(row)

    torch.save(tensor_payload, output_dir / "channel_scores.pt")
    write_csv(output_dir / "stability_summary.csv", stability_rows)

    stage_overlap_rows: List[Dict[str, Any]] = []
    stage_pairs = [
        ("predict_a", "predict_b"),
        ("predict_a", "predict_c"),
        ("predict_a", "predict_end"),
        ("predict_b", "predict_c"),
        ("predict_b", "predict_end"),
        ("predict_c", "predict_end"),
    ]
    layer_nodes = sorted({(layer, node) for _, layer, node in mean_scores})
    for layer, node in layer_nodes:
        available = [stage for stage in STAGES if (stage, layer, node) in mean_scores]
        if len(available) < 2:
            continue
        num_channels = mean_scores[(available[0], layer, node)].numel()
        specs = [(f"top{count}", min(count, num_channels)) for count in topk_counts]
        specs.extend(
            (f"top{fraction:g}", topk_count_from_fraction(num_channels, fraction))
            for fraction in topk_fractions
        )
        for lhs, rhs in stage_pairs:
            if (lhs, layer, node) not in mean_scores or (rhs, layer, node) not in mean_scores:
                continue
            row: Dict[str, Any] = {"layer": layer, "node": node, "lhs_stage": lhs, "rhs_stage": rhs}
            for label, k in specs:
                row[f"{label}_jaccard"] = jaccard_between_vectors(
                    mean_scores[(lhs, layer, node)],
                    mean_scores[(rhs, layer, node)],
                    k,
                )
            stage_overlap_rows.append(row)
    write_csv(output_dir / "stage_overlap_summary.csv", stage_overlap_rows)

    write_json(
        output_dir / "summary.json",
        {
            "config": vars(args),
            "resolved_model_path": model_path,
            "resolved_data_dir": data_dir,
            "num_samples": len(test_data),
            "num_skipped_samples": len(skipped_samples),
            "num_score_groups": len(capture.scores),
            "elapsed_seconds": elapsed,
            "skipped_samples": skipped_samples,
        },
    )
    print(f"Samples: {len(test_data)}")
    print(f"Skipped samples: {len(skipped_samples)}")
    print(f"Score groups: {len(capture.scores)}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Wrote channel stability to: {output_dir}")


if __name__ == "__main__":
    main()
