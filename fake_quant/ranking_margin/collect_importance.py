#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
for path in (PROJECT_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.tasks.v1_0.registry import get_loader, get_task_config

try:
    from .core import save_rank_importance
except ImportError:
    from fake_quant.ranking_margin.core import save_rank_importance
try:
    from fake_quant.smoothquant.collect_smooth_scales import slice_calibration_data
except ImportError:
    from ..smoothquant.collect_smooth_scales import slice_calibration_data


DEFAULT_MODEL_PATH = "/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B"
DEFAULT_DATA_DIR = "data/onerec_data/benchmark-data"
DEFAULT_OUTPUT_PATH = "fake_quant/ranking_margin/importances/onerec_ad_rank_importance_sample128.pt"
SID_TOKEN_PATTERN = re.compile(r"<s_[abc]_\d+>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ranking-margin channel importance for OneRec AD prompts.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--sample_size", default=128, type=int)
    parser.add_argument(
        "--sample_offset",
        default=0,
        type=int,
        help="Start offset within the split. Use 1000 to avoid overlap with eval sample1000.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--target_regex", default=None)
    parser.add_argument("--skip_regex", default=None)
    parser.add_argument("--collect_lm_head", action="store_true")
    parser.add_argument("--negative_rank", type=int, default=32, help="Boundary negative rank for token-level margin.")
    parser.add_argument("--margin_tau", type=float, default=0.0)
    parser.add_argument("--loss", default="softplus", choices=["softplus", "hinge"])
    parser.add_argument("--eta_eps", type=float, default=1e-3)
    parser.add_argument("--max_prompt_tokens", type=int, default=512)
    parser.add_argument("--max_sid_tokens", type=int, default=3)
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


def resolve_input_device(model: torch.nn.Module, fallback: str) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        for device in hf_device_map.values():
            if isinstance(device, str) and device not in {"cpu", "disk"}:
                return torch.device(device)
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def load_ad_data(
    tokenizer: Any,
    data_dir: str,
    split: str,
    sample_size: int,
    sample_offset: int = 0,
) -> Dict[str, Dict[str, Any]]:
    loader = get_loader(
        task_name="ad",
        data_dir=data_dir,
        enable_thinking=False,
        tokenizer=tokenizer,
    )
    if sample_offset > 0:
        data = loader.load_data(split=split, sample_size="full")
        return slice_calibration_data(
            data,
            sample_size=sample_size,
            sample_offset=sample_offset,
        )
    return loader.load_data(split=split, sample_size=sample_size)


def should_collect_module(
    name: str,
    *,
    collect_lm_head: bool,
    target_pattern: re.Pattern[str] | None,
    skip_pattern: re.Pattern[str] | None,
) -> bool:
    child_name = name.rsplit(".", 1)[-1]
    if not collect_lm_head and (child_name == "lm_head" or name == "lm_head"):
        return False
    if skip_pattern is not None and skip_pattern.search(name) is not None:
        return False
    if target_pattern is not None and target_pattern.search(name) is None:
        return False
    return True


def extract_sid_token_ids(tokenizer: Any, ground_truth: str, max_sid_tokens: int) -> list[int]:
    match = re.search(r"<\|sid_begin\|>(.*?)<\|sid_end\|>", ground_truth)
    if not match:
        return []
    sid_text = match.group(1)
    token_texts = SID_TOKEN_PATTERN.findall(sid_text)
    token_ids = []
    for token_text in token_texts[:max_sid_tokens]:
        ids = tokenizer.encode(token_text, add_special_tokens=False)
        if len(ids) != 1:
            return []
        token_ids.append(ids[0])
    return token_ids


def encode_prompt_ids(
    tokenizer: Any,
    prompt: str,
    prompt_token: str,
    max_prompt_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    if prompt_token and not prompt.endswith(prompt_token):
        prompt = prompt + prompt_token
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if max_prompt_tokens > 0 and prompt_ids.numel() > max_prompt_tokens:
        prompt_ids = prompt_ids[-max_prompt_tokens:]
    return prompt_ids.to(device)


def token_margin_loss(
    logits: torch.Tensor,
    target_token_id: int,
    *,
    negative_rank: int,
    margin_tau: float,
    loss_type: str,
    eta_eps: float,
) -> torch.Tensor:
    step_logits = logits.float()
    positive = step_logits[target_token_id]
    negative_logits = step_logits.clone()
    negative_logits[target_token_id] = -torch.inf
    k = min(max(negative_rank, 1), negative_logits.numel() - 1)
    negative = torch.topk(negative_logits, k=k).values[-1]
    margin = positive - negative
    eta = 1.0 / (torch.abs(margin.detach()) + eta_eps)
    if loss_type == "hinge":
        return eta * torch.relu(torch.as_tensor(margin_tau, device=logits.device) - margin)
    return eta * F.softplus(torch.as_tensor(margin_tau, device=logits.device) - margin)


def install_activation_hooks(
    model: nn.Module,
    *,
    collect_lm_head: bool,
    target_regex: str | None,
    skip_regex: str | None,
) -> tuple[dict[str, torch.Tensor], list[Any]]:
    activations: dict[str, torch.Tensor] = {}
    handles = []
    target_pattern = re.compile(target_regex) if target_regex else None
    skip_pattern = re.compile(skip_regex) if skip_regex else None

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

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if should_collect_module(
            name,
            collect_lm_head=collect_lm_head,
            target_pattern=target_pattern,
            skip_pattern=skip_pattern,
        ):
            handles.append(module.register_forward_pre_hook(make_hook(name)))

    return activations, handles


def install_embedding_grad_anchor(model: nn.Module) -> Any:
    embedding = model.get_input_embeddings()

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        return output.detach().requires_grad_(True)

    return embedding.register_forward_hook(hook)


def collect_rank_importance(
    model: nn.Module,
    tokenizer: Any,
    test_data: Dict[str, Dict[str, Any]],
    *,
    input_device: torch.device,
    prompt_token: str,
    collect_lm_head: bool,
    target_regex: str | None,
    skip_regex: str | None,
    negative_rank: int,
    margin_tau: float,
    loss_type: str,
    eta_eps: float,
    max_prompt_tokens: int,
    max_sid_tokens: int,
) -> Dict[str, torch.Tensor]:
    for param in model.parameters():
        param.requires_grad_(False)

    activations, handles = install_activation_hooks(
        model,
        collect_lm_head=collect_lm_head,
        target_regex=target_regex,
        skip_regex=skip_regex,
    )
    embedding_handle = install_embedding_grad_anchor(model)
    importance_sums: Dict[str, torch.Tensor] = {}
    importance_counts: Dict[str, int] = {}
    skipped = 0

    try:
        for sample in tqdm(test_data.values(), desc="Collect ranking-margin importance"):
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
                        token_id,
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

            for name, activation in activations.items():
                grad = activation.grad
                if grad is None:
                    continue
                reduce_dims = tuple(range(activation.ndim - 1))
                score = (activation.detach().float().abs() * grad.detach().float().abs()).mean(
                    dim=reduce_dims
                ).cpu()
                if name in importance_sums:
                    importance_sums[name] += score
                    importance_counts[name] += 1
                else:
                    importance_sums[name] = score
                    importance_counts[name] = 1

            del output, logits, loss
            model.zero_grad(set_to_none=True)
    finally:
        embedding_handle.remove()
        for handle in handles:
            handle.remove()

    rank_importance = {
        name: score / max(importance_counts[name], 1) for name, score in importance_sums.items()
    }
    if not rank_importance:
        raise RuntimeError(
            "No ranking importance stats were collected. "
            f"Skipped samples: {skipped}. Check ground_truth SID format and target/skip regex."
        )
    print(f"Skipped samples without usable SID targets: {skipped}")
    return rank_importance


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

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

    rank_importance = collect_rank_importance(
        model,
        tokenizer,
        test_data,
        input_device=input_device,
        prompt_token=prompt_token,
        collect_lm_head=args.collect_lm_head,
        target_regex=args.target_regex,
        skip_regex=args.skip_regex,
        negative_rank=args.negative_rank,
        margin_tau=args.margin_tau,
        loss_type=args.loss,
        eta_eps=args.eta_eps,
        max_prompt_tokens=args.max_prompt_tokens,
        max_sid_tokens=args.max_sid_tokens,
    )

    save_rank_importance(
        args.output_path,
        rank_importance=rank_importance,
        metadata={
            "model_path": args.model_path,
            "data_dir": args.data_dir,
            "split": args.split,
            "sample_size": args.sample_size,
            "sample_offset": args.sample_offset,
            "sample_range": [args.sample_offset, args.sample_offset + args.sample_size],
            "dtype": args.dtype,
            "seed": args.seed,
            "target_regex": args.target_regex,
            "skip_regex": args.skip_regex,
            "collect_lm_head": args.collect_lm_head,
            "negative_rank": args.negative_rank,
            "margin_tau": args.margin_tau,
            "loss": args.loss,
            "eta_eps": args.eta_eps,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_sid_tokens": args.max_sid_tokens,
        },
    )
    print(f"Saved ranking-margin importance to: {args.output_path}")
    print(f"Collected Linear modules: {len(rank_importance)}")


if __name__ == "__main__":
    main()
