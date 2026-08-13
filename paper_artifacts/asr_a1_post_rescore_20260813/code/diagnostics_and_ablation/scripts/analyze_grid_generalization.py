"""Analyze rerank grid generalization from validation to test.

The test grid is used only for analysis. Do not use this script to choose
official test parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-grid", required=True, help="Validation grid_metrics.jsonl")
    parser.add_argument("--test-grid", default="", help="Optional test grid_metrics.jsonl for analysis only")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k-values", default="5,10,20,50")
    parser.add_argument("--stable-top-k", type=int, default=20)
    parser.add_argument("--overfit-val-top-k", type=int, default=20)
    parser.add_argument("--overfit-test-rank-min", type=int, default=100)
    parser.add_argument("--overfit-wer-gap", type=float, default=0.001)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def param_key(record: Dict[str, Any]) -> str:
    payload = {
        "a": float(record.get("a", 0.0)),
        "b": float(record.get("b", 0.0)),
        "c": float(record.get("c", 0.0)),
        "d": float(record.get("d", 0.0)),
        "extra_weights": record.get("extra_weights") or {},
        "pruning": record.get("pruning") or {},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def compact_params(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "a": record.get("a"),
        "b": record.get("b"),
        "c": record.get("c"),
        "d": record.get("d"),
        "extra_weights": record.get("extra_weights") or {},
        "pruning": record.get("pruning") or {},
    }


def ranked(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(records, key=lambda row: (float(row.get("wer", math.inf)), float(row.get("cer", math.inf))))


def rank_map(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return {param_key(record): idx for idx, record in enumerate(ranked(records), start=1)}


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return float(num / (den_x * den_y))


def distribution(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    sorted_values = sorted(float(value) for value in values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        median = sorted_values[mid]
    else:
        median = 0.5 * (sorted_values[mid - 1] + sorted_values[mid])
    return {
        "count": len(sorted_values),
        "min": min(sorted_values),
        "mean": statistics.fmean(sorted_values),
        "median": median,
        "max": max(sorted_values),
    }


def merged_rows(
    val_records: Sequence[Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    test_by_key = {param_key(record): record for record in test_records}
    val_ranks = rank_map(val_records)
    test_ranks = rank_map(test_records)
    rows: List[Dict[str, Any]] = []
    for val_record in val_records:
        key = param_key(val_record)
        test_record = test_by_key.get(key)
        row: Dict[str, Any] = {
            **compact_params(val_record),
            "param_key": key,
            "val_rank": val_ranks.get(key),
            "val_wer": val_record.get("wer"),
            "val_cer": val_record.get("cer"),
            "val_bootstrap_selection_score": val_record.get("bootstrap_selection_score"),
            "val_bootstrap_wer_mean": val_record.get("bootstrap_wer_mean"),
            "val_bootstrap_wer_std": val_record.get("bootstrap_wer_std"),
        }
        if test_record is not None:
            row.update(
                {
                    "test_rank": test_ranks.get(key),
                    "test_wer": test_record.get("wer"),
                    "test_cer": test_record.get("cer"),
                    "test_minus_val_wer": float(test_record.get("wer", 0.0)) - float(val_record.get("wer", 0.0)),
                }
            )
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_grid = read_jsonl(Path(args.val_grid))
    if not val_grid:
        raise ValueError(f"No validation grid rows loaded from {args.val_grid}")
    test_grid = read_jsonl(Path(args.test_grid)) if args.test_grid else []

    val_ranked = ranked(val_grid)
    summary: Dict[str, Any] = {
        "val_grid": str(Path(args.val_grid)),
        "test_grid": str(Path(args.test_grid)) if args.test_grid else "",
        "val_rows": len(val_grid),
        "test_rows": len(test_grid),
        "note": "Test grid is analysis-only and must not be used for official parameter selection.",
        "val_best": {**compact_params(val_ranked[0]), "wer": val_ranked[0].get("wer"), "cer": val_ranked[0].get("cer")},
    }

    if test_grid:
        test_ranked = ranked(test_grid)
        val_ranks = rank_map(val_grid)
        test_ranks = rank_map(test_grid)
        common_keys = sorted(set(val_ranks) & set(test_ranks))
        val_rank_values = [float(val_ranks[key]) for key in common_keys]
        test_rank_values = [float(test_ranks[key]) for key in common_keys]
        merged = merged_rows(val_grid, test_grid)
        common_merged = [row for row in merged if row.get("test_rank") is not None]
        summary.update(
            {
                "test_best": {
                    **compact_params(test_ranked[0]),
                    "wer": test_ranked[0].get("wer"),
                    "cer": test_ranked[0].get("cer"),
                },
                "common_params": len(common_keys),
                "val_rank_vs_test_rank_spearman": pearson(val_rank_values, test_rank_values),
                "val_topk_test_wer_distribution": {},
            }
        )
        for k in parse_ints(args.top_k_values):
            top_keys = [param_key(record) for record in val_ranked[:k]]
            test_wers = [
                float(next(record for record in test_grid if param_key(record) == key).get("wer"))
                for key in top_keys
                if key in test_ranks
            ]
            summary["val_topk_test_wer_distribution"][str(k)] = distribution(test_wers)

        stable = [
            row
            for row in common_merged
            if int(row["val_rank"]) <= args.stable_top_k and int(row["test_rank"]) <= args.stable_top_k
        ]
        overfit = [
            row
            for row in common_merged
            if int(row["val_rank"]) <= args.overfit_val_top_k
            and (
                int(row["test_rank"]) >= args.overfit_test_rank_min
                or float(row.get("test_minus_val_wer", 0.0)) >= args.overfit_wer_gap
            )
        ]
        stable = sorted(stable, key=lambda row: (int(row["val_rank"]), int(row["test_rank"])))
        overfit = sorted(overfit, key=lambda row: (int(row["val_rank"]), -float(row.get("test_minus_val_wer", 0.0))))
        summary["stable_top_params_count"] = len(stable)
        summary["overfit_params_count"] = len(overfit)

        write_csv(output_dir / "val_test_grid_common.csv", common_merged)
        write_jsonl(output_dir / "stable_top_params.jsonl", stable)
        write_jsonl(output_dir / "overfit_params.jsonl", overfit)
    else:
        summary["val_top20"] = [
            {**compact_params(record), "wer": record.get("wer"), "cer": record.get("cer")}
            for record in val_ranked[:20]
        ]

    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
