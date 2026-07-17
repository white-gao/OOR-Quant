"""Entry point with robust handling of benchmark ``_total_time`` fields."""

from __future__ import annotations

from typing import Any, Mapping

from . import analyze_sid_stage_probe as _analysis


def metric_block(eval_json: Mapping[str, Any]) -> Mapping[str, Any]:
    model_entries = [value for key, value in eval_json.items() if key != "_total_time"]
    if len(model_entries) != 1:
        raise ValueError("Could not identify one model entry in eval_results.json")
    task_entries = [value for key, value in model_entries[0].items() if key != "_total_time"]
    if len(task_entries) != 1:
        raise ValueError("Could not identify one task entry in eval_results.json")
    split_entries = [value for key, value in task_entries[0].items() if key != "_total_time"]
    if len(split_entries) != 1:
        raise ValueError("Could not identify one split entry in eval_results.json")
    return split_entries[0]


_analysis.metric_block = metric_block


if __name__ == "__main__":
    _analysis.main()
