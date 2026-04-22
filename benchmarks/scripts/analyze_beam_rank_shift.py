#!/usr/bin/env python3
"""
Compare per-sample beam ranking changes between baseline and QDQ runs.

The script reads recommendation `test_generated.json` files, not
`eval_results.json`, because beam/rank analysis needs per-sample generations and
cum_logprobs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_BASELINE = Path("results/v1.0/results_OneRec-1.7B_ad_sample_1000")
DEFAULT_COMPARES = [
    "mlp-all=results/v1.0/results_OneRec-1.7B-fp8e4m3-mlp-all_ad_sample_1000",
    "block-linears=results/v1.0/results_OneRec-1.7B-fp8e4m3-block-linears_ad_sample_1000",
]
DEFAULT_OUTPUT_PREFIX = Path("results/v1.0/beam_rank_ad_sample_1000")
DEFAULT_K = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze beam/rank shifts from recommendation test_generated.json files."
    )
    parser.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE),
        help="Baseline result directory or test_generated.json path.",
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=None,
        help=(
            "Comparison result directory or test_generated.json path. "
            "Use label=path to control the output label. Can be repeated."
        ),
    )
    parser.add_argument("--task", default="ad")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top_k", type=int, default=DEFAULT_K)
    parser.add_argument("--output_prefix", default=str(DEFAULT_OUTPUT_PREFIX))
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_generation_path(path_like: str, task: str, split: str) -> Path:
    path = Path(path_like)
    if path.is_file():
        return path

    pattern = f"*/{task}/{split}_generated.json"
    matches = sorted(path.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No {split}_generated.json found under {path} with pattern {pattern}")
    raise ValueError(f"Multiple generation files found under {path}: {matches}")


def parse_compare_spec(spec: str, task: str, split: str) -> Tuple[str, Path]:
    if "=" in spec:
        label, path_str = spec.split("=", 1)
        label = label.strip()
        path_str = path_str.strip()
    else:
        path_str = spec.strip()
        label = label_from_path(Path(path_str))

    if not label:
        raise ValueError(f"Empty comparison label in spec: {spec!r}")
    return label, resolve_generation_path(path_str, task, split)


def label_from_path(path: Path) -> str:
    name = path.name
    if name == "test_generated.json":
        parts = path.parts
        if len(parts) >= 3:
            return parts[-3]
    if name.startswith("results_"):
        name = name[len("results_") :]
    suffixes = ["_ad_sample_1000", "_ad_full"]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    prefix = "OneRec-1.7B-fp8e4m3-"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    return name


def extract_sids_from_answer(answer: str) -> List[str]:
    sids: List[str] = []
    for part in str(answer).split("<|sid_begin|>"):
        if "<|sid_end|>" in part:
            sid = part.split("<|sid_end|>", 1)[0].strip()
            if sid and sid not in sids:
                sids.append(sid)
    return sids


def extract_sid_from_generation(generation: str) -> str:
    generation = str(generation).strip()
    if "</think>" in generation:
        generation = generation.split("</think>")[-1].strip()

    if "<|sid_begin|>" in generation:
        for part in generation.split("<|sid_begin|>"):
            if "<|sid_end|>" in part:
                sid = part.split("<|sid_end|>", 1)[0].strip()
                if sid:
                    return sid
            elif part.strip():
                return part.strip()
    return generation


def normalize_sids(generations: Sequence[Any], k: int) -> List[str]:
    return [extract_sid_from_generation(gen) for gen in generations[:k]]


def parse_pid_values(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        value = [value]

    pids: List[int] = []
    for item in value:
        try:
            pid = int(item)
        except (TypeError, ValueError):
            continue
        if pid != 0 and pid not in pids:
            pids.append(pid)
    return pids


def normalize_pids(sample: Dict[str, Any], k: int) -> List[int]:
    pids = sample.get("pid_generations", [])
    result: List[int] = []
    for pid in pids[:k]:
        try:
            value = int(pid)
        except (TypeError, ValueError):
            value = -1
        result.append(value)
    return result


def first_rank(candidates: Sequence[Any], targets: Set[Any], k: int) -> Optional[int]:
    if not targets:
        return None
    for idx, item in enumerate(candidates[:k], start=1):
        if item in targets:
            return idx
    return None


def hit_count(candidates: Sequence[Any], targets: Set[Any], k: int, ignore_values: Set[Any]) -> int:
    seen = {item for item in candidates[:k] if item not in ignore_values}
    return len(seen & targets)


def set_overlap(a: Sequence[Any], b: Sequence[Any], k: int, ignore_values: Set[Any]) -> Tuple[int, float]:
    set_a = {item for item in a[:k] if item not in ignore_values}
    set_b = {item for item in b[:k] if item not in ignore_values}
    if not set_a and not set_b:
        return 0, 1.0
    union = set_a | set_b
    inter = set_a & set_b
    return len(inter), len(inter) / len(union) if union else 0.0


def margin(logprobs: Sequence[Any]) -> Optional[float]:
    if len(logprobs) < 2:
        return None
    try:
        return float(logprobs[0]) - float(logprobs[1])
    except (TypeError, ValueError):
        return None


def rank_delta(baseline_rank: Optional[int], compare_rank: Optional[int], missing_rank: int) -> int:
    return (compare_rank or missing_rank) - (baseline_rank or missing_rank)


def rank_status(baseline_rank: Optional[int], compare_rank: Optional[int]) -> str:
    baseline_hit = baseline_rank is not None
    compare_hit = compare_rank is not None
    if baseline_hit and compare_hit:
        return "stable_hit"
    if baseline_hit and not compare_hit:
        return "lost"
    if not baseline_hit and compare_hit:
        return "gained"
    return "both_miss"


def bool_value(sample: Dict[str, Any], key: str) -> Optional[bool]:
    value = sample.get(key)
    if isinstance(value, bool):
        return value
    return None


def numeric_value(sample: Dict[str, Any], key: str) -> Optional[float]:
    value = sample.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def rate(values: Iterable[bool]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def build_sample_row(
    label: str,
    sample_id: str,
    baseline_sample: Dict[str, Any],
    compare_sample: Dict[str, Any],
    k: int,
) -> Dict[str, Any]:
    missing_rank = k + 1

    baseline_sids = normalize_sids(baseline_sample.get("generations", []), k)
    compare_sids = normalize_sids(compare_sample.get("generations", []), k)
    baseline_pids = normalize_pids(baseline_sample, k)
    compare_pids = normalize_pids(compare_sample, k)

    target_sids = extract_sids_from_answer(baseline_sample.get("ground_truth", ""))
    target_sid_set = set(target_sids)
    first_target_sid = target_sids[0] if target_sids else ""

    metadata = baseline_sample.get("metadata", {})
    target_pids = parse_pid_values(metadata.get("answer_pid", metadata.get("answer_iid")))
    target_pid_set = set(target_pids)
    first_target_pid = target_pids[0] if target_pids else None

    baseline_best_sid_rank = first_rank(baseline_sids, target_sid_set, k)
    compare_best_sid_rank = first_rank(compare_sids, target_sid_set, k)
    baseline_first_sid_rank = first_rank(baseline_sids, {first_target_sid} if first_target_sid else set(), k)
    compare_first_sid_rank = first_rank(compare_sids, {first_target_sid} if first_target_sid else set(), k)

    baseline_best_pid_rank = first_rank(baseline_pids, target_pid_set, k)
    compare_best_pid_rank = first_rank(compare_pids, target_pid_set, k)
    baseline_first_pid_rank = first_rank(baseline_pids, {first_target_pid} if first_target_pid else set(), k)
    compare_first_pid_rank = first_rank(compare_pids, {first_target_pid} if first_target_pid else set(), k)

    sid_overlap_count, sid_overlap_jaccard = set_overlap(baseline_sids, compare_sids, k, {""})
    pid_overlap_count, pid_overlap_jaccard = set_overlap(baseline_pids, compare_pids, k, {-1, 0})

    baseline_margin = margin(baseline_sample.get("logprobs", []))
    compare_margin = margin(compare_sample.get("logprobs", []))
    margin_delta = None
    if baseline_margin is not None and compare_margin is not None:
        margin_delta = compare_margin - baseline_margin

    row: Dict[str, Any] = {
        "comparison": label,
        "sample_id": sample_id,
        "row_index": metadata.get("row_index", ""),
        "uid": metadata.get("uid", ""),
        "ground_truth_sid_count": len(target_sids),
        "ground_truth_pid_count": len(target_pids),
        "first_target_sid": first_target_sid,
        "first_target_pid": first_target_pid if first_target_pid is not None else "",
        "target_sids_json": json.dumps(target_sids, ensure_ascii=False),
        "target_pids_json": json.dumps(target_pids, ensure_ascii=False),
        "baseline_top1_sid": baseline_sids[0] if baseline_sids else "",
        "compare_top1_sid": compare_sids[0] if compare_sids else "",
        "sid_top1_changed": (baseline_sids[:1] != compare_sids[:1]),
        "baseline_top1_pid": baseline_pids[0] if baseline_pids else "",
        "compare_top1_pid": compare_pids[0] if compare_pids else "",
        "pid_top1_changed": (baseline_pids[:1] != compare_pids[:1]),
        "sid_topk_overlap_count": sid_overlap_count,
        "sid_topk_overlap_jaccard": sid_overlap_jaccard,
        "pid_topk_overlap_count": pid_overlap_count,
        "pid_topk_overlap_jaccard": pid_overlap_jaccard,
        "baseline_best_sid_rank": baseline_best_sid_rank or "",
        "compare_best_sid_rank": compare_best_sid_rank or "",
        "best_sid_rank_delta_missing_as_k_plus_1": rank_delta(
            baseline_best_sid_rank, compare_best_sid_rank, missing_rank
        ),
        "best_sid_status": rank_status(baseline_best_sid_rank, compare_best_sid_rank),
        "baseline_first_sid_rank": baseline_first_sid_rank or "",
        "compare_first_sid_rank": compare_first_sid_rank or "",
        "first_sid_rank_delta_missing_as_k_plus_1": rank_delta(
            baseline_first_sid_rank, compare_first_sid_rank, missing_rank
        ),
        "first_sid_status": rank_status(baseline_first_sid_rank, compare_first_sid_rank),
        "baseline_best_pid_rank": baseline_best_pid_rank or "",
        "compare_best_pid_rank": compare_best_pid_rank or "",
        "best_pid_rank_delta_missing_as_k_plus_1": rank_delta(
            baseline_best_pid_rank, compare_best_pid_rank, missing_rank
        ),
        "best_pid_status": rank_status(baseline_best_pid_rank, compare_best_pid_rank),
        "baseline_first_pid_rank": baseline_first_pid_rank or "",
        "compare_first_pid_rank": compare_first_pid_rank or "",
        "first_pid_rank_delta_missing_as_k_plus_1": rank_delta(
            baseline_first_pid_rank, compare_first_pid_rank, missing_rank
        ),
        "first_pid_status": rank_status(baseline_first_pid_rank, compare_first_pid_rank),
        "baseline_sid_hit_count": hit_count(baseline_sids, target_sid_set, k, {""}),
        "compare_sid_hit_count": hit_count(compare_sids, target_sid_set, k, {""}),
        "baseline_pid_hit_count": hit_count(baseline_pids, target_pid_set, k, {-1, 0}),
        "compare_pid_hit_count": hit_count(compare_pids, target_pid_set, k, {-1, 0}),
        "baseline_top1_top2_margin": baseline_margin,
        "compare_top1_top2_margin": compare_margin,
        "top1_top2_margin_delta": margin_delta,
    }

    metric_keys = [
        "pass@1",
        f"pass@{k}",
        "position1_pass@1",
        f"position1_pass@{k}",
        "recall@1",
        f"recall@{k}",
        "pid_pass@1",
        f"pid_pass@{k}",
        "pid_position1_pass@1",
        f"pid_position1_pass@{k}",
        "pid_recall@1",
        f"pid_recall@{k}",
    ]
    for key in metric_keys:
        row[f"baseline_{key}"] = baseline_sample.get(key, "")
        row[f"compare_{key}"] = compare_sample.get(key, "")

    return row


def summarize_rows(label: str, rows: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "comparison": label,
        "samples": len(rows),
        "sid_top1_changed_pct": rate(row["sid_top1_changed"] for row in rows) * 100.0,
        "pid_top1_changed_pct": rate(row["pid_top1_changed"] for row in rows) * 100.0,
        "sid_topk_overlap_count_avg": mean(row["sid_topk_overlap_count"] for row in rows),
        "sid_topk_overlap_jaccard_avg": mean(row["sid_topk_overlap_jaccard"] for row in rows),
        "pid_topk_overlap_count_avg": mean(row["pid_topk_overlap_count"] for row in rows),
        "pid_topk_overlap_jaccard_avg": mean(row["pid_topk_overlap_jaccard"] for row in rows),
        "baseline_margin_avg": mean(row["baseline_top1_top2_margin"] for row in rows),
        "compare_margin_avg": mean(row["compare_top1_top2_margin"] for row in rows),
        "margin_delta_avg": mean(row["top1_top2_margin_delta"] for row in rows),
    }

    for metric in [
        "pass@1",
        f"pass@{k}",
        "position1_pass@1",
        f"position1_pass@{k}",
        "recall@1",
        f"recall@{k}",
        "pid_pass@1",
        f"pid_pass@{k}",
        "pid_position1_pass@1",
        f"pid_position1_pass@{k}",
        "pid_recall@1",
        f"pid_recall@{k}",
    ]:
        baseline_values: List[Optional[float]] = []
        compare_values: List[Optional[float]] = []
        for row in rows:
            baseline_key = f"baseline_{metric}"
            compare_key = f"compare_{metric}"
            if isinstance(row.get(baseline_key), bool):
                baseline_values.append(1.0 if row[baseline_key] else 0.0)
            else:
                baseline_values.append(numeric_or_none(row.get(baseline_key)))
            if isinstance(row.get(compare_key), bool):
                compare_values.append(1.0 if row[compare_key] else 0.0)
            else:
                compare_values.append(numeric_or_none(row.get(compare_key)))

        baseline_mean = mean(baseline_values)
        compare_mean = mean(compare_values)
        summary[f"baseline_{metric}"] = baseline_mean
        summary[f"compare_{metric}"] = compare_mean
        if baseline_mean is not None and compare_mean is not None:
            summary[f"{metric}_delta"] = compare_mean - baseline_mean

    for field in ["best_sid_status", "first_sid_status", "best_pid_status", "first_pid_status"]:
        counts = status_counts(row[field] for row in rows)
        for status, count in counts.items():
            summary[f"{field}_{status}"] = count
            summary[f"{field}_{status}_pct"] = count / len(rows) * 100.0 if rows else 0.0

    for field in [
        "best_sid_rank_delta_missing_as_k_plus_1",
        "first_sid_rank_delta_missing_as_k_plus_1",
        "best_pid_rank_delta_missing_as_k_plus_1",
        "first_pid_rank_delta_missing_as_k_plus_1",
    ]:
        summary[f"{field}_avg"] = mean(row[field] for row in rows)
        summary[f"{field}_abs_avg"] = mean(abs(float(row[field])) for row in rows)

    return summary


def numeric_or_none(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def status_counts(values: Iterable[str]) -> Dict[str, int]:
    counts = {"stable_hit": 0, "lost": 0, "gained": 0, "both_miss": 0}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary_rows: List[Dict[str, Any]], top_k: int) -> None:
    columns = [
        "comparison",
        "samples",
        "sid_top1_changed_pct",
        "sid_topk_overlap_jaccard_avg",
        "best_sid_status_lost",
        "best_sid_status_gained",
        f"pass@{top_k}_delta",
        f"recall@{top_k}_delta",
        "pid_top1_changed_pct",
        "pid_topk_overlap_jaccard_avg",
        "best_pid_status_lost",
        "best_pid_status_gained",
        f"pid_pass@{top_k}_delta",
        f"pid_recall@{top_k}_delta",
        "margin_delta_avg",
    ]

    with path.open("w", encoding="utf-8") as f:
        f.write("# Beam Rank Shift Summary\n\n")
        f.write(
            "This analysis compares per-sample `test_generated.json` files. "
            "`lost` means baseline hit at top-k but the comparison run missed; "
            "`gained` means the comparison run hit when baseline missed. "
            "Rank deltas treat a missing target as `k + 1`.\n\n"
        )
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in summary_rows:
            f.write("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |\n")


def main() -> None:
    args = parse_args()
    baseline_path = resolve_generation_path(args.baseline, args.task, args.split)
    compare_specs = args.compare if args.compare is not None else DEFAULT_COMPARES
    compares = [parse_compare_spec(spec, args.task, args.split) for spec in compare_specs]
    output_prefix = Path(args.output_prefix)

    baseline_data = load_json(baseline_path)
    baseline_samples = baseline_data.get("samples", {})
    if not baseline_samples:
        raise ValueError(f"No samples found in baseline file: {baseline_path}")

    all_sample_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for label, compare_path in compares:
        compare_data = load_json(compare_path)
        compare_samples = compare_data.get("samples", {})
        common_ids = sorted(set(baseline_samples) & set(compare_samples), key=lambda x: int(x) if x.isdigit() else x)
        if not common_ids:
            raise ValueError(f"No overlapping sample ids between {baseline_path} and {compare_path}")

        rows = [
            build_sample_row(
                label=label,
                sample_id=sample_id,
                baseline_sample=baseline_samples[sample_id],
                compare_sample=compare_samples[sample_id],
                k=args.top_k,
            )
            for sample_id in common_ids
        ]
        all_sample_rows.extend(rows)
        summary_rows.append(summarize_rows(label, rows, args.top_k))

    sample_columns = list(all_sample_rows[0].keys())
    summary_columns = list(summary_rows[0].keys())

    sample_csv = output_prefix.with_name(output_prefix.name + "_samples.csv")
    summary_csv = output_prefix.with_name(output_prefix.name + "_summary.csv")
    summary_md = output_prefix.with_name(output_prefix.name + "_summary.md")

    write_csv(sample_csv, all_sample_rows, sample_columns)
    write_csv(summary_csv, summary_rows, summary_columns)
    write_markdown(summary_md, summary_rows, args.top_k)

    print(f"Baseline: {baseline_path}")
    for label, compare_path in compares:
        print(f"Compare {label}: {compare_path}")
    print(f"Wrote samples CSV: {sample_csv}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote summary Markdown: {summary_md}")


if __name__ == "__main__":
    main()
