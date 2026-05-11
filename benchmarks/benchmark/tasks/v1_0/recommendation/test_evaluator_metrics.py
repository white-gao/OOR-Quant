from __future__ import annotations

from benchmark.tasks.v1_0.recommendation.evaluator import RecommendationEvaluator


def test_recommendation_metrics_use_intermediate_k_values_without_position_metrics() -> None:
    evaluator = RecommendationEvaluator(
        samples={},
        task_config={
            "evaluation_config": {
                "k_values": [1, 4, 8, 16, 32],
                "evaluation_mode": "both",
            }
        },
    )

    assert evaluator.required_metrics == [
        "pass@1",
        "pass@4",
        "pass@8",
        "pass@16",
        "pass@32",
        "recall@1",
        "recall@4",
        "recall@8",
        "recall@16",
        "recall@32",
        "pid_pass@1",
        "pid_pass@4",
        "pid_pass@8",
        "pid_pass@16",
        "pid_pass@32",
        "pid_recall@1",
        "pid_recall@4",
        "pid_recall@8",
        "pid_recall@16",
        "pid_recall@32",
    ]


def test_calculated_metric_order_groups_pass_then_recall() -> None:
    evaluator = RecommendationEvaluator(samples={})
    k_values = [1, 4, 8, 16, 32]

    metrics = evaluator._calculate_metrics_from_counts(
        pass_counts={k: k for k in k_values},
        recall_sums={k: float(k) / 2 for k in k_values},
        total_samples=32,
        k_values=k_values,
        prefix="pid_",
    )

    assert list(metrics) == [
        "pid_pass@1",
        "pid_pass@4",
        "pid_pass@8",
        "pid_pass@16",
        "pid_pass@32",
        "pid_recall@1",
        "pid_recall@4",
        "pid_recall@8",
        "pid_recall@16",
        "pid_recall@32",
    ]
    assert "pid_position1_pass@1" not in metrics


def test_overall_metric_order_puts_accuracy_metrics_before_metadata() -> None:
    evaluator = RecommendationEvaluator(
        samples={
            "s1": {
                "generations": ["<|sid_begin|>A<|sid_end|>"],
                "ground_truth": "<|sid_begin|>A<|sid_end|>",
            }
        },
        task_config={
            "evaluation_config": {
                "k_values": [1, 4],
                "evaluation_mode": "sid",
                "select_k": "first_k",
            }
        },
        overwrite=True,
    )

    metrics, per_sample_metrics = evaluator.evaluate()

    assert list(metrics)[:4] == ["pass@1", "pass@4", "recall@1", "recall@4"]
    assert list(metrics)[4:] == ["total_samples", "select_k_strategy", "evaluation_mode"]
    assert "position1_pass@1" not in metrics
    assert "position1_pass@1" not in per_sample_metrics["s1"]


def test_position_metrics_are_kept_for_debug_but_not_returned_as_outputs() -> None:
    evaluator = RecommendationEvaluator(
        samples={
            "s1": {
                "generations": ["<|sid_begin|>A<|sid_end|>"],
                "ground_truth": "<|sid_begin|>A<|sid_end|>",
                "metadata": {},
            }
        },
        debug=True,
    )

    _pass_counts, _recall_sums, per_sample_metrics, debug_info = evaluator._evaluate_single_mode(
        k_values=[1],
        evaluation_mode="sid",
        select_k_strategy="first_k",
    )

    assert "position1_pass@1" not in per_sample_metrics["s1"]
    assert debug_info["passed_samples"][0]["position1_pass_results"] == {
        "position1_pass@1": True
    }
