#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .apply import (
    BaselineQuantSummary,
    apply_baseline_w8a8,
    apply_learnable_lwt,
    export_learned_quant_params,
    freeze_learnable_lwt,
    learned_quantized_module_from_params,
)
from .calibrate_m1_lwt import Batch, CalibrationHistory, calibrate_block_mse
from .modules import BaselineFakeQuantLinear, LearnableFakeQuantLinear
from .quant import ActQuant


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
for path in (PROJECT_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark import Benchmark  # noqa: E402
from benchmark.tasks.v1_0.registry import get_loader, get_task_config  # noqa: E402


DEFAULT_MODEL_PATH = "/home/guowei/OneRec-1.7B"
DEFAULT_DATA_DIR = "data/onerec_data/benchmark-data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OneRec AD evaluation with baseline, M1 LWT, or M2 LWT+LET FP8 W+A fake quantization."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--eval_data_dir", default=None)
    parser.add_argument("--output_dir", default="fake_quant_learnable/results/ptq_ad")
    parser.add_argument("--mode", default="m1_lwt", choices=["baseline_w8a8", "m1_lwt", "m2_lwt_let"])
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--calib_sample_size", default="128")
    parser.add_argument("--calib_offset", type=int, default=0)
    parser.add_argument("--eval_sample_size", default="128")
    parser.add_argument("--eval_offset", type=int, default=0)
    parser.add_argument("--calib_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--layers", default="last:1", help='Layer spec: "all", "last:K", "0,2-4".')
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init_clip_multiplier", type=float, default=1.0)
    parser.add_argument("--act_quant", default="per_token", choices=["none", "per_token"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--num_beams", type=int, default=32)
    parser.add_argument("--num_return_sequences", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--calib_only", action="store_true")
    parser.add_argument("--save_model_state", action="store_true")
    parser.add_argument("--save_quant_params", dest="save_quant_params", action="store_true", default=True)
    parser.add_argument("--no_save_quant_params", dest="save_quant_params", action="store_false")
    parser.add_argument("--load_quant_params", default=None)
    parser.add_argument("--skip_calibration", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def parse_sample_size(value: Any) -> Any:
    if value is None or value == "":
        return None
    if value == "full":
        return "full"
    return int(value)


def resolve_repo_path(path: str | os.PathLike[str]) -> Path:
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_input_device(model: nn.Module, fallback: str) -> torch.device:
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


def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise AttributeError("Could not find transformer layers at model.model.layers or model.layers.")
    if not isinstance(layers, nn.ModuleList):
        raise TypeError(f"Expected layers to be nn.ModuleList, got {type(layers)!r}.")
    return layers


def parse_layer_indices(spec: str, *, num_layers: int) -> list[int]:
    spec = spec.strip()
    if spec == "all":
        return list(range(num_layers))
    if spec.startswith("last:"):
        count = int(spec.split(":", 1)[1])
        if count <= 0 or count > num_layers:
            raise ValueError(f"Invalid last layer count: {count}")
        return list(range(num_layers - count, num_layers))

    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid layer range: {part}")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    deduped = sorted(set(indices))
    if not deduped:
        raise ValueError("No layer indices selected.")
    for idx in deduped:
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {idx} out of range for {num_layers} layers.")
    return deduped


def load_ad_data(
    tokenizer: Any,
    data_dir: str,
    split: str,
    sample_size: Any,
    sample_offset: int = 0,
) -> dict[str, dict[str, Any]]:
    if sample_offset < 0:
        raise ValueError(f"sample_offset must be non-negative, got {sample_offset}")

    loader_sample_size = sample_size
    if isinstance(sample_size, int):
        loader_sample_size = sample_size + sample_offset

    loader = get_loader(
        task_name="ad",
        data_dir=data_dir,
        enable_thinking=False,
        tokenizer=tokenizer,
    )
    data = loader.load_data(split=split, sample_size=loader_sample_size)
    if sample_offset == 0 and not isinstance(sample_size, int):
        return data

    items = list(data.items())
    if sample_offset:
        if sample_offset >= len(items):
            raise ValueError(
                f"sample_offset={sample_offset} leaves no samples after loading {len(items)} rows."
            )
        items = items[sample_offset:]
    if isinstance(sample_size, int):
        items = items[:sample_size]
    return dict(items)


def iter_batches(items: Sequence[Any], batch_size: int) -> Iterator[list[Any]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def format_prompt(prompt: str, prompt_token: str) -> str:
    if prompt_token and not prompt.endswith(prompt_token):
        return prompt + prompt_token
    return prompt


def build_model_batches(
    *,
    tokenizer: Any,
    prompts: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    old_padding_side = getattr(tokenizer, "padding_side", None)
    if old_padding_side is not None:
        tokenizer.padding_side = "left"
    try:
        batches: list[dict[str, torch.Tensor]] = []
        for prompt_batch in iter_batches(list(prompts), batch_size):
            encoded = tokenizer(prompt_batch, return_tensors="pt", padding=True)
            batches.append(
                {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in encoded.items()
                }
            )
        return batches
    finally:
        if old_padding_side is not None:
            tokenizer.padding_side = old_padding_side


def capture_layer_input_batches(
    *,
    model: nn.Module,
    layer: nn.Module,
    model_batches: Iterable[Mapping[str, Any]],
) -> list[Batch]:
    captured: list[Batch] = []

    def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        captured.append((_detach_tree(args), _detach_tree(kwargs)))

    handle = layer.register_forward_pre_hook(hook, with_kwargs=True)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in model_batches:
                try:
                    model(**batch, use_cache=False)
                except TypeError:
                    model(**batch)
    finally:
        handle.remove()
        model.train(was_training)
    return captured


def calibrate_model_layers_m1(
    *,
    model: nn.Module,
    model_batches: Sequence[Mapping[str, Any]],
    layer_indices: Sequence[int],
    steps: int,
    lr: float,
    act_quant: ActQuant,
    init_clip_multiplier: float,
    enable_let: bool = False,
    learned_quant_params: MutableMapping[int, dict[str, Any]] | None = None,
) -> dict[int, CalibrationHistory]:
    layers = get_transformer_layers(model)
    histories: dict[int, CalibrationHistory] = {}
    for layer_idx in layer_indices:
        teacher_block = layers[layer_idx]
        captured = capture_layer_input_batches(
            model=model,
            layer=teacher_block,
            model_batches=model_batches,
        )
        quant_block = copy.deepcopy(teacher_block)
        if isinstance(quant_block, nn.Linear):
            quant_block = LearnableFakeQuantLinear(
                quant_block,
                act_quant=act_quant,
                init_clip_multiplier=init_clip_multiplier,
                enable_let=enable_let,
            )
        else:
            apply_learnable_lwt(
                quant_block,
                act_quant=act_quant,
                init_clip_multiplier=init_clip_multiplier,
                enable_let=enable_let,
            )
        history = calibrate_block_mse(
            teacher_block=teacher_block,
            quant_block=quant_block,
            batches=captured,
            steps=steps,
            lr=lr,
        )
        if learned_quant_params is not None:
            learned_quant_params[layer_idx] = export_learned_quant_params(quant_block)
        if isinstance(quant_block, LearnableFakeQuantLinear):
            quant_block = quant_block.to_frozen()
        else:
            freeze_learnable_lwt(quant_block)
        layers[layer_idx] = quant_block
        histories[layer_idx] = history
        label = "M2" if enable_let else "M1"
        print(
            f"[{label}] layer={layer_idx} initial_loss={history.initial_loss:.6g} "
            f"final_loss={history.final_loss:.6g}"
        )
    return histories


def apply_baseline_layers(
    *,
    model: nn.Module,
    layer_indices: Sequence[int],
    act_quant: ActQuant,
) -> dict[int, BaselineQuantSummary]:
    """Apply min-max FP8 W+A fake quantization to selected transformer layers."""
    layers = get_transformer_layers(model)
    summaries: dict[int, BaselineQuantSummary] = {}
    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        if isinstance(layer, nn.Linear):
            layers[layer_idx] = BaselineFakeQuantLinear(layer, act_quant=act_quant)
            summary = BaselineQuantSummary(replaced_linears=1, skipped_linears=0)
        else:
            summary = apply_baseline_w8a8(layer, act_quant=act_quant)
        summaries[layer_idx] = summary
        print(
            f"[baseline_w8a8] layer={layer_idx} replaced_linears={summary.replaced_linears} "
            f"skipped_linears={summary.skipped_linears}"
        )
    return summaries


def apply_learned_quant_params_to_layers(model: nn.Module, payload: Mapping[str, Any]) -> list[int]:
    """Apply saved learned quant params to transformer layers and freeze wrappers."""
    layers_payload = payload.get("layers")
    if not isinstance(layers_payload, Mapping):
        raise TypeError("learned quant params payload must contain a layers mapping.")

    layers = get_transformer_layers(model)
    applied: list[int] = []
    for layer_key, layer_params in sorted(layers_payload.items(), key=lambda item: int(item[0])):
        layer_idx = int(layer_key)
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of range for {len(layers)} layers.")
        new_layer, _count = learned_quantized_module_from_params(layers[layer_idx], layer_params)
        layers[layer_idx] = new_layer
        applied.append(layer_idx)
    return applied


def build_learned_quant_params_payload(
    *,
    method: str,
    act_quant: ActQuant,
    layers: Mapping[int, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "method": method,
        "quant_format": "fp8_e4m3fn",
        "act_quant": act_quant,
        "layers": dict(layers),
    }


def decode_generations(tokenizer: Any, sequences: torch.Tensor, prompt_len: int) -> list[str]:
    generations = []
    for seq in sequences:
        generated_ids = seq[prompt_len:]
        generations.append(tokenizer.decode(generated_ids, skip_special_tokens=False))
    return generations


def generate_batch(
    *,
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    input_device: torch.device,
    args: argparse.Namespace,
) -> list[list[str]]:
    if not prompts:
        return []

    old_padding_side = getattr(tokenizer, "padding_side", None)
    if old_padding_side is not None:
        tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(list(prompts), return_tensors="pt", padding=True)
        inputs = {
            key: value.to(input_device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        prompt_len = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        if old_padding_side is not None:
            tokenizer.padding_side = old_padding_side

    output = output.detach().cpu()
    if output.shape[0] % len(prompts) != 0:
        raise RuntimeError(f"Unexpected generate output shape: {tuple(output.shape)}")
    returns_per_prompt = output.shape[0] // len(prompts)
    generations = []
    for batch_idx in range(len(prompts)):
        start = batch_idx * returns_per_prompt
        end = start + returns_per_prompt
        generations.append(decode_generations(tokenizer, output[start:end], prompt_len))
    return generations


def result_path(output_dir: str, model_name: str, split: str) -> Path:
    return resolve_repo_path(output_dir) / model_name / "ad" / f"{split}_generated.json"


def save_results(
    *,
    output_file: Path,
    model_name: str,
    split: str,
    test_data: Mapping[str, Mapping[str, Any]],
    generations: Mapping[str, list[str]],
    total_time: float,
    config: Mapping[str, Any],
) -> None:
    samples: dict[str, dict[str, Any]] = {}
    for sample_id, sample in test_data.items():
        item = {
            "prompt": sample.get("prompt", ""),
            "generations": generations.get(sample_id, []),
            "ground_truth": sample.get("ground_truth", ""),
        }
        if "metadata" in sample:
            item["metadata"] = sample["metadata"]
        samples[sample_id] = item

    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "task_name": "ad",
        "split": split,
        "total_time": total_time,
        "avg_time_per_sample": total_time / len(samples) if samples else 0.0,
        "quant_config": dict(config),
        "learnable_quant_config": dict(config),
        "samples": samples,
    }
    output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def maybe_evaluate(output_dir: str, data_dir: str, overwrite: bool) -> None:
    output_root = resolve_repo_path(output_dir)
    data_root = resolve_repo_path(data_dir)
    Benchmark.evaluate_dev(
        generation_results_dir=str(output_root),
        output_path=str(output_root / "eval_results.json"),
        data_dir=str(data_root),
        overwrite=overwrite,
        task_types=["ad"],
    )


def histories_to_jsonable(histories: Mapping[int, CalibrationHistory]) -> dict[str, Any]:
    return {
        str(layer_idx): {
            "initial_loss": history.initial_loss,
            "final_loss": history.final_loss,
            "losses": history.losses,
        }
        for layer_idx, history in histories.items()
    }


def summaries_to_jsonable(summaries: Mapping[int, BaselineQuantSummary]) -> dict[str, Any]:
    return {
        str(layer_idx): {
            "replaced_linears": summary.replaced_linears,
            "skipped_linears": summary.skipped_linears,
        }
        for layer_idx, summary in summaries.items()
    }


def _detach_tree(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().clone()
    if isinstance(obj, tuple):
        return tuple(_detach_tree(item) for item in obj)
    if isinstance(obj, list):
        return [_detach_tree(item) for item in obj]
    if isinstance(obj, Mapping):
        return {key: _detach_tree(value) for key, value in obj.items()}
    return obj


def main() -> None:
    args = parse_args()
    if args.calib_batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.calib_offset < 0 or args.eval_offset < 0:
        raise ValueError("Offsets must be non-negative.")
    set_seed(args.seed)

    model_name = args.model_name or Path(args.model_path.rstrip("/")).name
    output_file = result_path(args.output_dir, model_name, args.split)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and not args.overwrite and not args.calib_only:
        raise FileExistsError(f"Generation file exists: {output_file}. Use --overwrite.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
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

    layers = get_transformer_layers(model)
    layer_indices = parse_layer_indices(args.layers, num_layers=len(layers))
    histories: dict[int, CalibrationHistory] = {}
    baseline_summaries: dict[int, BaselineQuantSummary] = {}
    learned_quant_params: dict[int, dict[str, Any]] = {}
    quant_params_payload: dict[str, Any] | None = None
    quant_params_file: Path | None = None

    if args.load_quant_params:
        if args.mode not in {"m1_lwt", "m2_lwt_let"}:
            raise ValueError("--load_quant_params is only supported for learnable modes.")
        load_path = resolve_repo_path(args.load_quant_params)
        try:
            quant_params_payload = torch.load(load_path, map_location="cpu", weights_only=False)
        except TypeError:
            quant_params_payload = torch.load(load_path, map_location="cpu")
        if not isinstance(quant_params_payload, Mapping):
            raise TypeError(f"Expected quant params payload mapping, got {type(quant_params_payload)!r}.")
        payload_method = quant_params_payload.get("method")
        if payload_method is not None and payload_method != args.mode:
            raise ValueError(f"Loaded quant params method={payload_method!r} does not match mode={args.mode!r}.")
        layer_indices = apply_learned_quant_params_to_layers(model, quant_params_payload)
        print(f"[load_quant_params] applied layers={layer_indices} from {load_path}")
    elif args.skip_calibration:
        raise ValueError("--skip_calibration requires --load_quant_params.")
    elif args.mode in {"m1_lwt", "m2_lwt_let"}:
        calib_data = load_ad_data(
            tokenizer,
            str(resolve_repo_path(args.data_dir)),
            args.split,
            parse_sample_size(args.calib_sample_size),
            sample_offset=args.calib_offset,
        )
        calib_prompts = [format_prompt(sample["prompt"], prompt_token) for sample in calib_data.values()]
        calib_batches = build_model_batches(
            tokenizer=tokenizer,
            prompts=calib_prompts,
            batch_size=args.calib_batch_size,
            device=input_device,
        )
        histories = calibrate_model_layers_m1(
            model=model,
            model_batches=calib_batches,
            layer_indices=layer_indices,
            steps=args.steps,
            lr=args.lr,
            act_quant=args.act_quant,
            init_clip_multiplier=args.init_clip_multiplier,
            enable_let=args.mode == "m2_lwt_let",
            learned_quant_params=learned_quant_params,
        )
        quant_params_payload = build_learned_quant_params_payload(
            method=args.mode,
            act_quant=args.act_quant,
            layers=learned_quant_params,
        )
    elif args.mode == "baseline_w8a8":
        baseline_summaries = apply_baseline_layers(
            model=model,
            layer_indices=layer_indices,
            act_quant=args.act_quant,
        )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    if args.save_quant_params and learned_quant_params:
        quant_params_file = output_file.parent / f"{args.mode}_learned_quant_params.pt"
        torch.save(quant_params_payload, quant_params_file)

    config = {
        "method": args.mode,
        "layers": layer_indices,
        "steps": args.steps,
        "lr": args.lr,
        "init_clip_multiplier": args.init_clip_multiplier,
        "act_quant": args.act_quant,
        "calib_sample_size": args.calib_sample_size,
        "calib_offset": args.calib_offset,
        "calib_batch_size": args.calib_batch_size,
        "eval_sample_size": args.eval_sample_size,
        "eval_offset": args.eval_offset,
        "eval_batch_size": args.eval_batch_size,
        "dtype": args.dtype,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "save_quant_params": args.save_quant_params,
        "load_quant_params": args.load_quant_params,
        "skip_calibration": args.skip_calibration,
        "quant_params_file": None if quant_params_file is None else str(quant_params_file),
        "histories": histories_to_jsonable(histories),
        "baseline_summaries": summaries_to_jsonable(baseline_summaries),
    }
    config_filename = {
        "m1_lwt": "m1_calibration.json",
        "m2_lwt_let": "m2_calibration.json",
        "baseline_w8a8": "baseline_w8a8_config.json",
    }[args.mode]
    (output_file.parent / config_filename).write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.save_model_state:
        torch.save(model.state_dict(), output_file.parent / f"{args.mode}_quantized_model_state.pt")
    if args.calib_only:
        return

    test_data = load_ad_data(
        tokenizer,
        str(resolve_repo_path(args.data_dir)),
        args.split,
        parse_sample_size(args.eval_sample_size),
        sample_offset=args.eval_offset,
    )
    test_items = list(test_data.items())
    generations: dict[str, list[str]] = {}
    start = time.time()
    for batch in tqdm(
        iter_batches(test_items, args.eval_batch_size),
        total=(len(test_items) + args.eval_batch_size - 1) // args.eval_batch_size,
        desc=f"{args.mode} AD generation",
    ):
        sample_ids = [sample_id for sample_id, _sample in batch]
        prompts = [format_prompt(sample["prompt"], prompt_token) for _sample_id, sample in batch]
        batch_generations = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            input_device=input_device,
            args=args,
        )
        for sample_id, sample_generations in zip(sample_ids, batch_generations):
            generations[sample_id] = sample_generations
    total_time = time.time() - start

    save_results(
        output_file=output_file,
        model_name=model_name,
        split=args.split,
        test_data=test_data,
        generations=generations,
        total_time=total_time,
        config=config,
    )
    if args.evaluate:
        maybe_evaluate(args.output_dir, args.eval_data_dir or args.data_dir, args.overwrite)


if __name__ == "__main__":
    main()
