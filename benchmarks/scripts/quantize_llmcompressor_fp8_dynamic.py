#!/usr/bin/env python3
"""Create a vLLM-compatible compressed checkpoint with llmcompressor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a HuggingFace checkpoint with llmcompressor."
    )
    parser.add_argument(
        "--model_path",
        default="/home/guowei/OneRec-1.7B/",
        help="Input HuggingFace model path.",
    )
    parser.add_argument(
        "--output_dir",
        default="models/OneRec-1.7B-FP8-DYNAMIC",
        help="Output directory for the compressed checkpoint.",
    )
    parser.add_argument(
        "--scheme",
        default="fp8_dynamic",
        choices=["fp8_dynamic", "fp8_static", "fp8_weight_only"],
        help="Built-in quantization scheme. Ignored when --config is set.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a JSON quantization config under quantization_configs/.",
    )
    parser.add_argument(
        "--targets",
        default="Linear",
        help="llmcompressor target modules. Default quantizes Linear modules.",
    )
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=["lm_head"],
        help="Module names to keep unquantized.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16"],
        help="Model loading precision used by llmcompressor.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the model.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only build and print the recipe without loading/quantizing the model.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Calibration dataset name or loader type, e.g. wikitext or json.",
    )
    parser.add_argument(
        "--dataset_config_name",
        default=None,
        help="Optional HuggingFace dataset config name.",
    )
    parser.add_argument(
        "--dataset_path",
        default=None,
        help="Optional local calibration data path for dataset loaders such as json.",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="Calibration split name, e.g. train.",
    )
    parser.add_argument(
        "--text_column",
        default="text",
        help="Text column used by the calibration dataset.",
    )
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=512,
        help="Number of samples used to calibrate static activation scales.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for calibration samples.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Calibration batch size.",
    )
    parser.add_argument(
        "--pipeline",
        default="independent",
        choices=["independent", "sequential", "datafree", "basic"],
        help="llmcompressor calibration pipeline.",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    if not args.config:
        return {
            "scheme": args.scheme,
            "targets": args.targets,
            "ignore": args.ignore,
        }

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    config.setdefault("scheme", args.scheme)
    config.setdefault("targets", args.targets)
    config.setdefault("ignore", args.ignore)
    return config


def build_recipe(args: argparse.Namespace):
    import torch
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from llmcompressor.modifiers.quantization import QuantizationModifier

    def _torch_dtype(name: str | None):
        if name is None:
            return None
        dtype_map = {
            "float8_e4m3fn": torch.float8_e4m3fn,
            "int8": torch.int8,
            "uint8": torch.uint8,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if name not in dtype_map:
            raise ValueError(f"Unsupported dtype in quantization config: {name}")
        return dtype_map[name]

    def _build_quant_args(config: dict, *, default_strategy: str) -> QuantizationArgs:
        quant_type = config.get("type", "float")
        return QuantizationArgs(
            num_bits=config.get("num_bits", 8),
            type=quant_type,
            symmetric=config.get("symmetric", True),
            group_size=config.get("group_size"),
            strategy=config.get("strategy", default_strategy),
            block_structure=config.get("block_structure"),
            dynamic=config.get("dynamic", False),
            observer=config.get("observer", "memoryless_minmax"),
            scale_dtype=_torch_dtype(config.get("scale_dtype")),
            zp_dtype=_torch_dtype(config.get("zp_dtype", "float8_e4m3fn" if quant_type == "float" else "int8")),
        )

    config = load_config(args)
    scheme_name = config["scheme"]
    targets = config["targets"]
    ignore = config["ignore"]

    preset_schemes = {
        "fp8_dynamic": "FP8_DYNAMIC",
        "fp8_static": "FP8",
        "fp8_block": "FP8_BLOCK",
        "int8": "INT8",
        "w8a16": "W8A16",
        "w4a16": "W4A16",
        "w4a16_asym": "W4A16_ASYM",
    }
    if scheme_name in preset_schemes:
        return QuantizationModifier(
            targets=targets,
            scheme=preset_schemes[scheme_name],
            ignore=ignore,
        )

    custom_schemes = {
        "fp8_weight_only",
        "fp8_weight_channel_act_tensor_static",
    }
    if scheme_name not in custom_schemes:
        raise ValueError(f"Unsupported quantization scheme: {scheme_name}")

    weights_config = config.get("weights", {})
    weight_args = _build_quant_args(weights_config, default_strategy="channel")

    input_activations_config = config.get("input_activations")
    input_activations = None
    if input_activations_config:
        input_activations = _build_quant_args(
            input_activations_config,
            default_strategy="tensor",
        )

    scheme = QuantizationScheme(
        targets=[targets] if isinstance(targets, str) else targets,
        weights=weight_args,
        input_activations=input_activations,
        output_activations=None,
    )
    return QuantizationModifier(
        config_groups={"group_0": scheme},
        ignore=ignore,
    )


def needs_calibration(config: dict) -> bool:
    for key in ("input_activations", "output_activations"):
        quant_args = config.get(key)
        if quant_args and not quant_args.get("dynamic", False):
            return True
    return False


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from llmcompressor import oneshot

    recipe = build_recipe(args)
    config = load_config(args)
    print("Input model:", args.model_path)
    print("Output dir:", output_dir)
    print("Config:", args.config or "<built-in>")
    print("Scheme:", config["scheme"])
    print("Recipe:", recipe)
    if args.dry_run:
        return

    if needs_calibration(config) and not (args.dataset or args.dataset_path):
        raise ValueError(
            "This config uses static activation quantization and needs calibration data. "
            "Pass --dataset for a HuggingFace dataset, or --dataset json --dataset_path <file> "
            "for a local JSON/JSONL file."
        )

    dataset = args.dataset
    dataset_path = args.dataset_path
    splits = args.splits
    if args.dataset == "json" and args.dataset_path:
        from datasets import load_dataset

        dataset_path = str(Path(args.dataset_path).expanduser().resolve())
        dataset = load_dataset("json", data_files=dataset_path, split=args.splits or "train")
        splits = None
        dataset_path = None
        print(f"Loaded local JSON calibration dataset: {len(dataset)} samples")

    calibration_kwargs = {
        "dataset": dataset,
        "dataset_config_name": args.dataset_config_name,
        "dataset_path": dataset_path,
        "splits": splits,
        "text_column": args.text_column,
        "num_calibration_samples": args.num_calibration_samples,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "pipeline": args.pipeline,
    }
    calibration_kwargs = {
        key: value for key, value in calibration_kwargs.items() if value is not None
    }

    oneshot(
        model=args.model_path,
        recipe=recipe,
        output_dir=str(output_dir),
        precision=args.dtype,
        trust_remote_code_model=args.trust_remote_code,
        save_compressed=True,
        **calibration_kwargs,
    )

    config_path = output_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        print("Saved quantization_config:")
        print(json.dumps(config.get("quantization_config", {}), indent=2))
    else:
        print("Warning: config.json was not found in output_dir")


if __name__ == "__main__":
    main()
