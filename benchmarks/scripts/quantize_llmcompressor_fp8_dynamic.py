#!/usr/bin/env python3
"""Create a vLLM-compatible FP8 checkpoint with llmcompressor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a HuggingFace checkpoint with llmcompressor FP8."
    )
    parser.add_argument(
        "--model_path",
        default="/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B",
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
        choices=["fp8_dynamic", "fp8_weight_only", "fp8_static"],
        help=(
            "Quantization scheme. fp8_dynamic uses the llmcompressor "
            "FP8_DYNAMIC preset; fp8_static uses the FP8 preset; "
            "fp8_weight_only uses a custom FP8 per-channel weight-only scheme."
        ),
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
    return parser.parse_args()


def build_recipe(args: argparse.Namespace):
    import torch
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from llmcompressor.modifiers.quantization import QuantizationModifier

    if args.scheme == "fp8_dynamic":
        return QuantizationModifier(
            targets=args.targets,
            scheme="FP8_DYNAMIC",
            ignore=args.ignore,
        )

    if args.scheme == "fp8_static":
        return QuantizationModifier(
            targets=args.targets,
            scheme="FP8",
            ignore=args.ignore,
        )

    weight_args = QuantizationArgs(
        num_bits=8,
        type="float",
        symmetric=True,
        strategy="channel",
        dynamic=False,
        observer="memoryless_minmax",
        zp_dtype=torch.float8_e4m3fn,
    )
    scheme = QuantizationScheme(
        targets=[args.targets],
        weights=weight_args,
        input_activations=None,
        output_activations=None,
    )
    return QuantizationModifier(
        config_groups={"group_0": scheme},
        ignore=args.ignore,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from llmcompressor import oneshot

    recipe = build_recipe(args)
    print("Input model:", args.model_path)
    print("Output dir:", output_dir)
    print("Scheme:", args.scheme)
    print("Recipe:", recipe)
    if args.dry_run:
        return

    oneshot(
        model=args.model_path,
        recipe=recipe,
        output_dir=str(output_dir),
        precision=args.dtype,
        trust_remote_code_model=args.trust_remote_code,
        save_compressed=True,
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
