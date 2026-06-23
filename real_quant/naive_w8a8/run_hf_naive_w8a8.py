from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from real_quant.full_precision.generator import HFFullPrecisionGenerator, dtype_from_name
from real_quant.full_precision.results import build_generation_payload, save_generation_payload
from real_quant.full_precision.run_hf_baseline import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_PATH,
    RECOMMENDATION_TASKS,
    BatchSizeArg,
    build_output_samples,
    load_task_data,
    maybe_evaluate,
    parse_batch_size_arg,
    parse_sample_size,
    resolve_batch_size,
    resolve_repo_path,
    result_path,
)

from .apply import NaiveW8A8Summary, apply_naive_w8a8
from .modules import FP8_MAX, require_fp8_runtime


DEFAULT_OUTPUT_DIR = "real_quant/naive_w8a8/results"


class HFNaiveW8A8Generator(HFFullPrecisionGenerator):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        model_name: str,
        device: str | torch.device,
        num_params: float | None,
        quant_summary: NaiveW8A8Summary,
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            device=device,
            num_params=num_params,
        )
        self.quant_summary = quant_summary

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        attn_implementation: str | None = None,
        target_regex: str | None = None,
        skip_regex: str | None = None,
        use_fast_accum: bool = False,
    ) -> "HFNaiveW8A8Generator":
        require_real_fp8_device(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": dtype_from_name(dtype),
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        num_params = float(sum(p.numel() for p in model.parameters()))
        output_dtype = dtype_from_name(dtype)
        if output_dtype not in (torch.bfloat16, torch.float16):
            output_dtype = torch.bfloat16
        quant_summary = apply_naive_w8a8(
            model,
            target_regex=target_regex,
            skip_regex=skip_regex,
            output_dtype=output_dtype,
            use_fast_accum=use_fast_accum,
        )
        model.to(device)
        model.eval()
        model_name = f"{Path(model_path.rstrip('/')).name}-real-naive-w8a8"
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            device=device,
            num_params=num_params,
            quant_summary=quant_summary,
        )


def require_real_fp8_device(device: str | torch.device) -> None:
    require_fp8_runtime()
    device_obj = torch.device(device)
    if device_obj.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real naive W8A8 requires a CUDA device with torch._scaled_mm FP8 support.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenOneRec-style real naive W8A8 HuggingFace baseline using torch._scaled_mm."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task", default="ad", choices=RECOMMENDATION_TASKS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample_size", default="full")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch_size", type=parse_batch_size_arg, default=1)
    parser.add_argument("--num_beams", type=int, default=32)
    parser.add_argument("--num_return_sequences", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--prompt_token", default=None)
    parser.add_argument("--output_scores", action="store_true")
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--target_regex", default=None)
    parser.add_argument("--skip_regex", default=None)
    parser.add_argument("--use_fast_accum", action="store_true")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


def _batch_size_value(value: int | BatchSizeArg) -> int | str:
    return int(value) if value != "auto" else "auto"


def main() -> None:
    args = parse_args()
    batch_size, batch_size_config = resolve_batch_size(
        args.batch_size,
        device=args.device,
        model_path=args.model_path,
        task=args.task,
    )
    if batch_size_config.get("auto_batch_size"):
        print(
            "[hf_naive_w8a8] auto batch_size="
            f"{batch_size} (total_memory_gb={batch_size_config.get('auto_batch_total_memory_gb'):.2f}, "
            f"model_size_b={batch_size_config.get('auto_batch_model_size_billions')}, task={args.task})"
        )

    generator = HFNaiveW8A8Generator.from_pretrained(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        target_regex=args.target_regex,
        skip_regex=args.skip_regex,
        use_fast_accum=args.use_fast_accum,
    )
    model_name = str(generator)
    output_file = result_path(args.output_dir, model_name, args.task, args.split)
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Generation file exists: {output_file}. Use --overwrite.")

    from benchmark.tasks.v1_0.registry import get_task_config

    task_config = get_task_config(args.task)
    generation_config = dict(task_config.get("generation_config", {}))
    prompt_token = args.prompt_token
    if prompt_token is None:
        prompt_token = generation_config.get("prompt_token", "<|sid_begin|>")

    sample_size = parse_sample_size(args.sample_size)
    test_data = load_task_data(
        task_name=args.task,
        data_dir=str(resolve_repo_path(args.data_dir)),
        tokenizer=generator.tokenizer,
        split=args.split,
        sample_size=sample_size,
    )
    prompts = {sample_id: sample["prompt"] for sample_id, sample in test_data.items()}
    generations, _ = generator.generate(
        prompts,
        prompt_token=prompt_token,
        batch_size=batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        output_scores=args.output_scores,
    )

    config = {
        "backend": "hf_real_naive_w8a8_scaled_mm",
        "reference": "OpenOneRec HuggingFace generate with nn.Linear replaced by torch._scaled_mm FP8 wrappers",
        "task": args.task,
        "split": args.split,
        "data_dir": args.data_dir,
        "sample_size": args.sample_size,
        "dtype": args.dtype,
        "device": args.device,
        "batch_size": batch_size,
        "requested_batch_size": _batch_size_value(args.batch_size),
        **batch_size_config,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "max_new_tokens": args.max_new_tokens,
        "prompt_token": prompt_token,
        "output_scores": args.output_scores,
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": args.trust_remote_code,
        "fp8_dtype": "float8_e4m3fn",
        "kernel": "torch._scaled_mm",
        "weight_quant": "per_output_channel_absmax",
        "activation_quant": "per_token_dynamic_absmax",
        "activation_quant_sharing": "qkv_and_gate_up_shared_input",
        "qmax": FP8_MAX,
        "target_regex": args.target_regex,
        "skip_regex": args.skip_regex,
        "skip_module_names": ["lm_head"],
        "use_fast_accum": args.use_fast_accum,
        "replaced_linears": generator.quant_summary.replaced_linears,
        "skipped_linears": generator.quant_summary.skipped_linears,
        "shared_attention_modules": generator.quant_summary.shared_attention_modules,
        "shared_mlp_modules": generator.quant_summary.shared_mlp_modules,
    }
    samples = build_output_samples(test_data=test_data, generations=generations)
    payload = build_generation_payload(
        model_name=model_name,
        task_name=args.task,
        split=args.split,
        samples=samples,
        latency_records=list(generator.latency_records.values()),
        config=config,
        hardware_info=generator.get_hardware_info(),
        num_params=generator.num_params,
    )
    save_generation_payload(payload, output_file)
    (output_file.parent / "hf_naive_w8a8_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Generation results saved to: {output_file}")
    print(
        "Latency summary: "
        f"generate_total={payload['latency']['generate_time_total']:.3f}s, "
        f"end_to_end_total={payload['latency']['end_to_end_time_total']:.3f}s, "
        f"avg_generate={payload['latency']['generate_time_avg']:.6f}s/sample"
    )
    print(
        "Quant summary: "
        f"replaced_linears={generator.quant_summary.replaced_linears}, "
        f"skipped_linears={generator.quant_summary.skipped_linears}, "
        f"shared_attention_modules={generator.quant_summary.shared_attention_modules}, "
        f"shared_mlp_modules={generator.quant_summary.shared_mlp_modules}"
    )

    if args.evaluate:
        maybe_evaluate(args.output_dir, args.data_dir, args.overwrite, task_name=args.task)


if __name__ == "__main__":
    main()
