"""Causal SID-generation-stage activation rescue probe.

This runner calibrates Plain GPTQ once, then evaluates W8A8 and selected
W8A16 rescue variants with identical quantized weights. It is intentionally
separate from the main quantization CLI.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Mapping

import torch
from tqdm.auto import tqdm

from real_quant.full_precision.generator import (
    HFFullPrecisionGenerator,
    append_prompt_token,
    build_hf_generation_kwargs,
)
from real_quant.full_precision.results import build_generation_payload, save_generation_payload
from real_quant.full_precision.run_hf_baseline import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_PATH,
    build_output_samples,
    load_task_data,
    maybe_evaluate,
    parse_sample_size,
    resolve_repo_path,
    result_path,
)

from .run_hf_naive_w8a8 import HFNaiveW8A8Generator
from .stage_probe_runtime import activate_stage_activation_rescue


DEFAULT_OUTPUT_DIR = "real_quant/naive_w8a8/results/probes/sid_generation_stage"
DEFAULT_VARIANTS = ("w8a8", "rescue_a", "rescue_b", "rescue_c", "rescue_all")
SID_TRIPLE = re.compile(r"<s_a_\d+><s_b_\d+><s_c_\d+>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-wise activation rescue for Plain GPTQ W8A8.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--task", default="ad")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample_size", default="1000")
    parser.add_argument("--gptq_calib_split", default="calib")
    parser.add_argument("--gptq_calib_sample_size", default="128")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num_beams", type=int, default=32)
    parser.add_argument("--num_return_sequences", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--trace_sample_size", type=int, default=128)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_variants(value: str) -> list[str]:
    variants = [item.strip() for item in value.split(",") if item.strip()]
    invalid = set(variants) - set(DEFAULT_VARIANTS)
    if invalid:
        raise ValueError(f"Unsupported variants: {sorted(invalid)}")
    if not variants:
        raise ValueError("At least one variant is required.")
    return variants


def stages_for_variant(variant: str) -> set[str]:
    mapping = {
        "w8a8": set(),
        "rescue_a": {"a"},
        "rescue_b": {"b"},
        "rescue_c": {"c"},
        "rescue_all": {"a", "b", "c"},
    }
    return mapping[variant]


@contextlib.contextmanager
def maybe_stage_rescue(model: torch.nn.Module, variant: str) -> Iterator[None]:
    stages = stages_for_variant(variant)
    if not stages:
        yield
        return
    with activate_stage_activation_rescue(model, stages):
        yield


def sid_triples(text: str) -> set[str]:
    return set(SID_TRIPLE.findall(text))


def sample_hit(sample: Mapping[str, Any]) -> bool:
    ground_truth = sid_triples(str(sample.get("ground_truth", "")))
    predictions: set[str] = set()
    for generation in sample.get("generations", []):
        predictions.update(sid_triples(str(generation)))
    return bool(ground_truth & predictions)


def paired_recovery(base_samples: Mapping[str, Mapping[str, Any]], rescue_samples: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    base_ids = set(base_samples)
    if base_ids != set(rescue_samples):
        raise ValueError("Base and rescue samples do not have identical sample IDs.")
    recovery_ids: list[str] = []
    regression_ids: list[str] = []
    both_hit = 0
    both_miss = 0
    for sample_id in sorted(base_ids, key=lambda value: int(value) if str(value).isdigit() else str(value)):
        base_hit = sample_hit(base_samples[sample_id])
        rescue_hit = sample_hit(rescue_samples[sample_id])
        if not base_hit and rescue_hit:
            recovery_ids.append(str(sample_id))
        elif base_hit and not rescue_hit:
            regression_ids.append(str(sample_id))
        elif base_hit:
            both_hit += 1
        else:
            both_miss += 1
    return {
        "base_hit_count": both_hit + len(regression_ids),
        "rescue_hit_count": both_hit + len(recovery_ids),
        "recovery_count": len(recovery_ids),
        "regression_count": len(regression_ids),
        "net_gain": len(recovery_ids) - len(regression_ids),
        "both_hit_count": both_hit,
        "both_miss_count": both_miss,
        "recovery_sample_ids": recovery_ids,
        "regression_sample_ids": regression_ids,
    }


def collect_beam_traces(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Mapping[str, str],
    prompt_token: str,
    generation_kwargs: Mapping[str, Any],
    variant: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Record final returned-beam prefixes and transition scores.

    HuggingFace's public generate output exposes final beam ancestry rather
    than every live beam at every step. The resulting prefix coverage is used
    as a comparable trajectory proxy, not as an exact beam-survival count.
    """
    records: list[dict[str, Any]] = []
    items = list(prompts.items())[: max(0, int(limit))]
    for sample_id, prompt in tqdm(items, desc=f"trace {variant}", unit="sample"):
        formatted = append_prompt_token(prompt, prompt_token)
        inputs = tokenizer(formatted, return_tensors="pt")
        inputs = {key: value.to(next(model.parameters()).device) if torch.is_tensor(value) else value for key, value in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode(), maybe_stage_rescue(model, variant):
            output = model.generate(**inputs, **dict(generation_kwargs))
        sequences = output.sequences.detach().cpu()
        transition = model.compute_transition_scores(
            output.sequences,
            output.scores,
            output.beam_indices,
            normalize_logits=False,
        ).detach().cpu()
        rows: list[dict[str, Any]] = []
        for row_index, sequence in enumerate(sequences):
            generated = sequence[prompt_len: prompt_len + int(generation_kwargs["max_new_tokens"])].tolist()
            prefixes = [tokenizer.decode(generated[: step + 1], skip_special_tokens=False) for step in range(len(generated))]
            rows.append(
                {
                    "generated_token_ids": generated,
                    "prefixes": prefixes,
                    "transition_scores": [float(value) for value in transition[row_index, : len(generated)].tolist()],
                    "sequence_score": float(getattr(output, "sequences_scores", torch.zeros(sequences.shape[0]))[row_index].detach().cpu().item()),
                }
            )
        records.append({"sample_id": str(sample_id), "variant": variant, "beams": rows})
    return records


def save_variant(
    *,
    generator: HFNaiveW8A8Generator,
    test_data: Mapping[str, Mapping[str, Any]],
    generations: Mapping[str, list[str]],
    output_root: Path,
    variant: str,
    config: Mapping[str, Any],
    overwrite: bool,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    samples = build_output_samples(test_data=dict(test_data), generations=dict(generations))
    model_name = f"{generator.model_name}-stage-{variant}"
    output_file = result_path(str(output_root), model_name, str(config["task"]), str(config["split"]))
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_file}. Use --overwrite.")
    payload = build_generation_payload(
        model_name=model_name,
        task_name=str(config["task"]),
        split=str(config["split"]),
        samples=samples,
        latency_records=list(generator.latency_records.values()),
        config=dict(config),
        hardware_info=generator.get_hardware_info(),
        num_params=generator.num_params,
    )
    save_generation_payload(payload, output_file)
    output_file.parent.joinpath("sid_stage_probe_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_file, samples


def main() -> None:
    args = parse_args()
    variants = parse_variants(args.variants)
    output_root = resolve_repo_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    generator = HFNaiveW8A8Generator.from_pretrained(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        weight_quant_mode="gptq",
        task=args.task,
        split=args.split,
        data_dir=args.data_dir,
        gptq_calib_split=args.gptq_calib_split,
        gptq_calib_sample_size=args.gptq_calib_sample_size,
    )
    from benchmark.tasks.v1_0.registry import get_task_config

    task_config = get_task_config(args.task)
    generation_config = dict(task_config.get("generation_config", {}))
    prompt_token = generation_config.get("prompt_token", "<|sid_begin|>")
    test_data = load_task_data(
        task_name=args.task,
        data_dir=str(resolve_repo_path(args.data_dir)),
        tokenizer=generator.tokenizer,
        split=args.split,
        sample_size=parse_sample_size(args.sample_size),
    )
    prompts = {sample_id: sample["prompt"] for sample_id, sample in test_data.items()}
    generation_kwargs = {
        "prompt_token": prompt_token,
        "batch_size": 1,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
    }
    trace_generation_kwargs = build_hf_generation_kwargs(
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        pad_token_id=generator.tokenizer.pad_token_id,
        eos_token_id=generator.tokenizer.eos_token_id,
        output_scores=True,
    )
    common_config = {
        "probe": "sid_generation_stage_activation_rescue",
        "probe_description": "Plain GPTQ FP8 weights fixed; selected stages bypass activation FP8-QDQ via W8A16.",
        "task": args.task,
        "split": args.split,
        "sample_size": args.sample_size,
        "data_dir": args.data_dir,
        "model_path": args.model_path,
        "weight_quant_mode": "gptq",
        "gptq_calib_split": args.gptq_calib_split,
        "gptq_calib_sample_size": args.gptq_calib_sample_size,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "max_new_tokens": args.max_new_tokens,
        "prompt_token": prompt_token,
        "trace_sample_size": args.trace_sample_size,
        "stage_definitions": {
            "a": "prefill final sid_begin position predicts SID-a",
            "b": "first decode token SID-a predicts SID-b",
            "c": "second decode token SID-b predicts SID-c",
        },
    }

    variant_samples: dict[str, dict[str, dict[str, Any]]] = {}
    outputs: dict[str, str] = {}
    for variant in variants:
        print(f"[sid_stage_probe] running {variant} stages={sorted(stages_for_variant(variant))}")
        with maybe_stage_rescue(generator.model, variant):
            generations, _ = generator.generate(prompts, **generation_kwargs)
        config = {**common_config, "variant": variant, "rescued_stages": sorted(stages_for_variant(variant))}
        variant_root = output_root / variant
        output_file, samples = save_variant(
            generator=generator,
            test_data=test_data,
            generations=generations,
            output_root=variant_root,
            variant=variant,
            config=config,
            overwrite=args.overwrite,
        )
        maybe_evaluate(str(variant_root), args.data_dir, args.overwrite, task_name=args.task)
        variant_samples[variant] = samples
        outputs[variant] = str(output_file)
        trace_records = collect_beam_traces(
            model=generator.model,
            tokenizer=generator.tokenizer,
            prompts=prompts,
            prompt_token=prompt_token,
            generation_kwargs=trace_generation_kwargs,
            variant=variant,
            limit=args.trace_sample_size,
        )
        trace_path = output_root / f"trace_{variant}.jsonl"
        with trace_path.open("w", encoding="utf-8") as handle:
            for record in trace_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if "w8a8" not in variant_samples:
        raise ValueError("The probe requires the w8a8 variant for paired recovery analysis.")
    paired = {
        variant: paired_recovery(variant_samples["w8a8"], samples)
        for variant, samples in variant_samples.items()
        if variant != "w8a8"
    }
    summary = {
        "outputs": outputs,
        "paired_recovery_vs_w8a8": paired,
        "notes": [
            "Rescue variants retain the same Plain GPTQ FP8 weights as W8A8.",
            "Beam traces contain final returned-beam prefixes and transition scores; they are a trajectory proxy, not all live beams.",
        ],
    }
    summary_path = output_root / "stage_probe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[sid_stage_probe] summary: {summary_path}")


if __name__ == "__main__":
    main()
