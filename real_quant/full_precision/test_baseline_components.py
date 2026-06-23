from __future__ import annotations

from real_quant.full_precision.generator import append_prompt_token, build_hf_generation_kwargs
from real_quant.full_precision.latency import LatencyRecord, aggregate_latency
from real_quant.full_precision.results import build_generation_payload
from real_quant.full_precision.run_hf_baseline import (
    batched_items,
    choose_auto_batch_size,
    infer_model_size_billions,
    parse_args,
    parse_batch_size_arg,
)


def test_append_prompt_token_adds_sid_begin_once() -> None:
    assert append_prompt_token("prompt", "<|sid_begin|>") == "prompt<|sid_begin|>"
    assert append_prompt_token("prompt<|sid_begin|>", "<|sid_begin|>") == "prompt<|sid_begin|>"
    assert append_prompt_token("prompt", None) == "prompt"


def test_build_hf_generation_kwargs_matches_recommendation_beam_defaults() -> None:
    kwargs = build_hf_generation_kwargs(
        max_new_tokens=3,
        num_beams=32,
        num_return_sequences=32,
        pad_token_id=0,
        eos_token_id=1,
    )

    assert kwargs == {
        "max_new_tokens": 3,
        "num_beams": 32,
        "num_return_sequences": 32,
        "do_sample": False,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "use_cache": True,
    }


def test_build_hf_generation_kwargs_can_enable_score_collection() -> None:
    kwargs = build_hf_generation_kwargs(
        max_new_tokens=3,
        num_beams=32,
        num_return_sequences=32,
        pad_token_id=0,
        eos_token_id=1,
        output_scores=True,
    )

    assert kwargs["return_dict_in_generate"] is True
    assert kwargs["output_scores"] is True


def test_aggregate_latency_reports_generate_and_end_to_end_stats() -> None:
    records = [
        LatencyRecord(
            sample_id="a",
            prompt_tokens=10,
            generated_sequences=32,
            generated_tokens=96,
            tokenize_time=0.1,
            generate_time=1.0,
            decode_time=0.2,
        ),
        LatencyRecord(
            sample_id="b",
            prompt_tokens=20,
            generated_sequences=32,
            generated_tokens=96,
            tokenize_time=0.2,
            generate_time=2.0,
            decode_time=0.3,
        ),
    ]

    summary = aggregate_latency(records)

    assert summary["num_samples"] == 2
    assert summary["total_prompt_tokens"] == 30
    assert summary["total_generated_tokens"] == 192
    assert summary["generate_time_total"] == 3.0
    assert summary["end_to_end_time_total"] == 3.8
    assert summary["generate_time_avg"] == 1.5
    assert summary["end_to_end_time_avg"] == 1.9


def test_generation_payload_keeps_openonerec_schema_and_latency_fields() -> None:
    records = [
        LatencyRecord(
            sample_id="sample-1",
            prompt_tokens=8,
            generated_sequences=2,
            generated_tokens=6,
            tokenize_time=0.01,
            generate_time=0.20,
            decode_time=0.03,
        )
    ]
    payload = build_generation_payload(
        model_name="OneRec-1.7B",
        task_name="ad",
        split="test",
        samples={
            "sample-1": {
                "prompt": "prompt<|sid_begin|>",
                "generations": ["<s_a_1><s_b_2><s_c_3>"],
                "ground_truth": "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>",
                "metadata": {"uid": "u1"},
            }
        },
        latency_records=records,
        config={"backend": "hf_full_precision"},
        hardware_info={"gpu_count": 1},
        num_params=123.0,
    )

    assert payload["model_name"] == "OneRec-1.7B"
    assert payload["task_name"] == "ad"
    assert payload["split"] == "test"
    assert payload["samples"]["sample-1"]["generations"] == ["<s_a_1><s_b_2><s_c_3>"]
    assert payload["samples"]["sample-1"]["metadata"] == {"uid": "u1"}
    assert payload["samples"]["sample-1"]["input_tokens"] == [8]
    assert payload["samples"]["sample-1"]["output_tokens"] == [6]
    assert payload["samples"]["sample-1"]["times"] == [0.2]
    assert payload["latency"]["generate_time_total"] == 0.2
    assert payload["quant_config"]["backend"] == "hf_full_precision"
    assert payload["hardware_info"] == {"gpu_count": 1}
    assert payload["num_params"] == 123.0


def test_batched_items_preserves_order_and_respects_batch_size() -> None:
    batches = list(batched_items([("a", 1), ("b", 2), ("c", 3)], batch_size=2))

    assert batches == [[("a", 1), ("b", 2)], [("c", 3)]]


def test_parse_batch_size_arg_accepts_auto_and_positive_int() -> None:
    assert parse_batch_size_arg("auto") == "auto"
    assert parse_batch_size_arg("4") == 4


def test_parse_args_defaults_to_batch_size_one(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_hf_baseline.py"])

    args = parse_args()

    assert args.batch_size == 1


def test_infer_model_size_billions_from_model_path() -> None:
    assert infer_model_size_billions("/models/OneRec-1.7B/") == 1.7
    assert infer_model_size_billions("/models/OneRec-8B") == 8.0
    assert infer_model_size_billions("/models/custom") is None


def test_choose_auto_batch_size_uses_large_memory_and_task_model_heuristics() -> None:
    assert choose_auto_batch_size(total_memory_gb=140.0, model_size_billions=1.7, task="ad") == 8
    assert choose_auto_batch_size(total_memory_gb=140.0, model_size_billions=1.7, task="video") == 4
    assert choose_auto_batch_size(total_memory_gb=140.0, model_size_billions=8.0, task="ad") == 4
    assert choose_auto_batch_size(total_memory_gb=140.0, model_size_billions=8.0, task="video") == 2
    assert choose_auto_batch_size(total_memory_gb=48.0, model_size_billions=8.0, task="ad") == 1
