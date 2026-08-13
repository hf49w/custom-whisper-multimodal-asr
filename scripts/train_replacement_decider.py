"""Train/select C2 replacement-decision rules on validation pairs.

The default method is a conservative rule grid. It does not learn ASR model
weights and never uses test data: it selects replacement gates and a decision
score on validation pairs only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a9_candidate_utils import read_jsonl
from c2_replacement_utils import (
    candidate_prediction_from_pair,
    current_prediction_from_pair,
    finite_float,
    finite_int,
    group_pairs,
    metrics_from_predictions,
    select_replacement,
    write_json,
    write_jsonl,
)
from visspeech_custom_whisper_utils import resolve_cross_platform_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-pairs-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=["rule_grid", "logistic", "gbdt"], default="rule_grid")
    parser.add_argument("--selection-metric", choices=["wer"], default="wer")
    parser.add_argument("--cost-false-replace", type=float, default=2.0)
    parser.add_argument("--cost-missed-replace", type=float, default=1.0)
    grid = parser.add_mutually_exclusive_group()
    grid.add_argument("--quick-grid", action="store_true", help="Use a compact grid. Default.")
    grid.add_argument("--full-grid", action="store_true", help="Use the full requested grid. This can be very slow.")
    parser.add_argument("--selection-method", choices=["best_val", "risk_adjusted"], default="risk_adjusted")
    parser.add_argument(
        "--max-grid-evals",
        type=int,
        default=0,
        help="Optional safety cap. 0 means evaluate the whole selected grid.",
    )
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def product_dict(grid: Mapping[str, Sequence[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid.keys())
    for values in itertools.product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def quick_condition_grid() -> Dict[str, Sequence[Any]]:
    return {
        "max_rank": [2, 3, 5, 10],
        "max_edit_distance": [1, 2, 3],
        "max_length_delta": [0, 1],
        "min_a1_delta": [-0.2, 0.0],
        "min_mbr_delta": [-0.2, 0.0],
        "min_teacher_delta": [0.0],
        "max_function_changes": [999],
        "max_article_changes": [999],
    }


def quick_weight_grid() -> Dict[str, Sequence[Any]]:
    return {
        "w_mbr": [0.5, 1.0],
        "w_a1": [0.0, 0.5],
        "w_a2": [0.0, 0.2],
        "w_a0": [0.0],
        "w_a5": [0.0],
        "w_len": [0.1],
        "w_edit": [0.1],
    }


def full_condition_grid() -> Dict[str, Sequence[Any]]:
    return {
        "max_rank": [2, 3, 5, 10, 20],
        "max_edit_distance": [1, 2, 3, 5, 10],
        "max_length_delta": [0, 1, 2, 3, 999],
        "min_a1_delta": [-0.5, -0.2, 0.0, 0.1, 0.2],
        "min_mbr_delta": [-0.5, -0.2, 0.0, 0.1, 0.2],
        "min_teacher_delta": [-1.0, -0.5, 0.0, 0.2, 0.5],
        "max_function_changes": [0, 1, 2, 999],
        "max_article_changes": [0, 1, 999],
    }


def full_weight_grid() -> Dict[str, Sequence[Any]]:
    return {
        "w_mbr": [0.0, 0.2, 0.5, 1.0],
        "w_a1": [0.0, 0.2, 0.5, 1.0],
        "w_a2": [0.0, 0.1, 0.2, 0.5],
        "w_a0": [0.0, 0.05, 0.1],
        "w_a5": [0.0, 0.05, 0.1],
        "w_len": [0.0, 0.05, 0.1, 0.2],
        "w_edit": [0.0, 0.05, 0.1, 0.2],
    }


def grid_records(*, quick: bool, max_grid_evals: int = 0) -> Iterable[Dict[str, Any]]:
    condition_grid = quick_condition_grid() if quick else full_condition_grid()
    weight_grid = quick_weight_grid() if quick else full_weight_grid()
    count = 0
    for condition in product_dict(condition_grid):
        for weights in product_dict(weight_grid):
            rule = dict(condition)
            rule.update(weights)
            count += 1
            if max_grid_evals > 0 and count > max_grid_evals:
                return
            yield rule


def grid_size(*, quick: bool) -> int:
    cond = quick_condition_grid() if quick else full_condition_grid()
    weights = quick_weight_grid() if quick else full_weight_grid()
    out = 1
    for values in list(cond.values()) + list(weights.values()):
        out *= len(values)
    return out


def group_has_benefit(group: Sequence[Mapping[str, Any]]) -> bool:
    return any(int(row.get("label_replace", 0)) == 1 for row in group)


def evaluate_rule(groups: Mapping[str, Sequence[Mapping[str, Any]]], rule: Mapping[str, Any]) -> Dict[str, Any]:
    predictions: List[Dict[str, Any]] = []
    replacements = improved = worsened = unchanged = 0
    false_replace = missed_replace = 0
    current_correct_replaced = 0
    current_wrong_improved = 0
    current_wrong_worsened = 0
    for group in groups.values():
        if not group:
            continue
        base_pair = group[0]
        selected, score = select_replacement(group, rule)
        if selected is None:
            predictions.append(current_prediction_from_pair(base_pair))
            if group_has_benefit(group):
                missed_replace += 1
            continue
        replacements += 1
        predictions.append(candidate_prediction_from_pair(selected, selector="replacement_decider_val", score=score))
        current_edits = finite_int(selected.get("current_word_edits"), 0)
        candidate_edits = finite_int(selected.get("candidate_word_edits"), 0)
        if current_edits == 0 and candidate_edits > 0:
            false_replace += 1
            current_correct_replaced += 1
        if candidate_edits < current_edits:
            improved += 1
            if current_edits > 0:
                current_wrong_improved += 1
        elif candidate_edits > current_edits:
            worsened += 1
            if current_edits > 0:
                current_wrong_worsened += 1
        else:
            unchanged += 1
        if group_has_benefit(group) and not (candidate_edits < current_edits):
            missed_replace += 1

    metrics = metrics_from_predictions(predictions)
    total = max(1, len(predictions))
    record: Dict[str, Any] = {
        "wer": float(metrics["wer"]),
        "cer": float(metrics["cer"]),
        "wer_percent": float(metrics["wer"]) * 100.0,
        "cer_percent": float(metrics["cer"]) * 100.0,
        "samples": len(predictions),
        "replacements_count": replacements,
        "improved_count": improved,
        "worsened_count": worsened,
        "unchanged_count": unchanged,
        "false_replace_count": false_replace,
        "missed_replace_count": missed_replace,
        "current_correct_but_replaced_count": current_correct_replaced,
        "current_wrong_and_improved_count": current_wrong_improved,
        "current_wrong_but_worsened_count": current_wrong_worsened,
        "replacement_rate": replacements / total,
        "improved_rate": improved / total,
        "worsened_rate": worsened / total,
        "false_replace_rate": false_replace / total,
        "missed_replace_rate": missed_replace / total,
    }
    record.update({key: value for key, value in rule.items()})
    record["risk_adjusted_score"] = (
        float(record["wer"]) + 0.5 * float(record["false_replace_rate"]) - 0.1 * float(record["improved_rate"])
    )
    return record


def prepare_fast_arrays(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    sample_to_index: Dict[str, int] = {}
    sample_ids: List[str] = []
    current_word_edits: List[int] = []
    current_char_edits: List[int] = []
    word_denoms: List[int] = []
    char_denoms: List[int] = []
    sample_has_benefit: List[bool] = []
    row_sample_index: List[int] = []

    fields = [
        "candidate_rank",
        "edit_distance_to_current",
        "abs_length_delta_words",
        "A1_logprob_z_delta",
        "mbr_z_delta",
        "teacher_z_delta_sum",
        "changed_function_count",
        "changed_article_count",
        "A2_logprob_z_delta",
        "A0_logprob_z_delta",
        "A5_logprob_z_delta",
        "candidate_word_edits",
        "candidate_char_edits",
    ]
    arrays: Dict[str, List[float]] = {field: [] for field in fields}

    for row in rows:
        sid = str(row.get("sample_id") or "")
        if sid not in sample_to_index:
            sample_to_index[sid] = len(sample_ids)
            sample_ids.append(sid)
            current_word_edits.append(finite_int(row.get("current_word_edits"), 0))
            current_char_edits.append(finite_int(row.get("current_char_edits"), 0))
            word_denoms.append(finite_int(row.get("word_denom"), 1))
            char_denoms.append(finite_int(row.get("char_denom"), 1))
            sample_has_benefit.append(False)
        sample_idx = sample_to_index[sid]
        row_sample_index.append(sample_idx)
        if int(row.get("label_replace", 0)) == 1:
            sample_has_benefit[sample_idx] = True
        for field in fields:
            arrays[field].append(finite_float(row.get(field), 0.0))

    out: Dict[str, Any] = {
        "sample_ids": sample_ids,
        "sample_index": np.asarray(row_sample_index, dtype=np.int32),
        "current_word_edits": np.asarray(current_word_edits, dtype=np.int32),
        "current_char_edits": np.asarray(current_char_edits, dtype=np.int32),
        "word_denoms": np.asarray(word_denoms, dtype=np.int32),
        "char_denoms": np.asarray(char_denoms, dtype=np.int32),
        "sample_has_benefit": np.asarray(sample_has_benefit, dtype=bool),
    }
    for field, values in arrays.items():
        dtype = np.float32
        if field in {"candidate_rank", "edit_distance_to_current", "abs_length_delta_words", "changed_function_count", "changed_article_count", "candidate_word_edits", "candidate_char_edits"}:
            dtype = np.int32
        out[field] = np.asarray(values, dtype=dtype)
    return out


def evaluate_rule_fast(data: Mapping[str, Any], rule: Mapping[str, Any]) -> Dict[str, Any]:
    sample_index = data["sample_index"]
    sample_count = int(len(data["sample_ids"]))
    mask = (
        (data["candidate_rank"] <= finite_float(rule.get("max_rank"), 0))
        & (data["edit_distance_to_current"] <= finite_float(rule.get("max_edit_distance"), 0))
        & (data["abs_length_delta_words"] <= finite_float(rule.get("max_length_delta"), 0))
        & (data["A1_logprob_z_delta"] >= finite_float(rule.get("min_a1_delta"), -math.inf))
        & (data["mbr_z_delta"] >= finite_float(rule.get("min_mbr_delta"), -math.inf))
        & (data["teacher_z_delta_sum"] >= finite_float(rule.get("min_teacher_delta"), -math.inf))
        & (data["changed_function_count"] <= finite_float(rule.get("max_function_changes"), math.inf))
        & (data["changed_article_count"] <= finite_float(rule.get("max_article_changes"), math.inf))
    )
    selected_rows = np.full(sample_count, -1, dtype=np.int64)
    selected_scores = np.full(sample_count, -np.inf, dtype=np.float32)
    masked_idx = np.nonzero(mask)[0]
    if masked_idx.size > 0:
        scores = (
            finite_float(rule.get("w_mbr"), 0.0) * data["mbr_z_delta"][masked_idx]
            + finite_float(rule.get("w_a1"), 0.0) * data["A1_logprob_z_delta"][masked_idx]
            + finite_float(rule.get("w_a2"), 0.0) * data["A2_logprob_z_delta"][masked_idx]
            + finite_float(rule.get("w_a0"), 0.0) * data["A0_logprob_z_delta"][masked_idx]
            + finite_float(rule.get("w_a5"), 0.0) * data["A5_logprob_z_delta"][masked_idx]
            - finite_float(rule.get("w_len"), 0.0) * data["abs_length_delta_words"][masked_idx]
            - finite_float(rule.get("w_edit"), 0.0) * data["edit_distance_to_current"][masked_idx]
        ).astype(np.float32)
        sample_ids = sample_index[masked_idx]
        # lexsort uses last key as primary: sample asc, score desc.
        order = np.lexsort((-scores, sample_ids))
        ordered_samples = sample_ids[order]
        first = np.r_[True, ordered_samples[1:] != ordered_samples[:-1]]
        winners = masked_idx[order[first]]
        winner_scores = scores[order[first]]
        winner_samples = sample_index[winners]
        selected_rows[winner_samples] = winners
        selected_scores[winner_samples] = winner_scores

    selected = selected_rows >= 0
    candidate_word = np.array(data["current_word_edits"], copy=True)
    candidate_char = np.array(data["current_char_edits"], copy=True)
    if np.any(selected):
        rows_selected = selected_rows[selected]
        candidate_word[selected] = data["candidate_word_edits"][rows_selected]
        candidate_char[selected] = data["candidate_char_edits"][rows_selected]

    current_word = data["current_word_edits"]
    replacements = int(selected.sum())
    improved_mask = selected & (candidate_word < current_word)
    worsened_mask = selected & (candidate_word > current_word)
    unchanged_mask = selected & (candidate_word == current_word)
    false_replace_mask = selected & (current_word == 0) & (candidate_word > 0)
    missed_mask = data["sample_has_benefit"] & ~improved_mask
    total = max(1, sample_count)
    wer = float(candidate_word.sum() / max(1, int(data["word_denoms"].sum())))
    cer = float(candidate_char.sum() / max(1, int(data["char_denoms"].sum())))
    record: Dict[str, Any] = {
        "wer": wer,
        "cer": cer,
        "wer_percent": wer * 100.0,
        "cer_percent": cer * 100.0,
        "samples": sample_count,
        "replacements_count": replacements,
        "improved_count": int(improved_mask.sum()),
        "worsened_count": int(worsened_mask.sum()),
        "unchanged_count": int(unchanged_mask.sum()),
        "false_replace_count": int(false_replace_mask.sum()),
        "missed_replace_count": int(missed_mask.sum()),
        "current_correct_but_replaced_count": int(false_replace_mask.sum()),
        "current_wrong_and_improved_count": int((improved_mask & (current_word > 0)).sum()),
        "current_wrong_but_worsened_count": int((worsened_mask & (current_word > 0)).sum()),
        "replacement_rate": replacements / total,
        "improved_rate": float(improved_mask.sum() / total),
        "worsened_rate": float(worsened_mask.sum() / total),
        "false_replace_rate": float(false_replace_mask.sum() / total),
        "missed_replace_rate": float(missed_mask.sum() / total),
        "mean_selected_score": float(np.mean(selected_scores[selected])) if replacements else 0.0,
    }
    record.update({key: value for key, value in rule.items()})
    record["risk_adjusted_score"] = (
        float(record["wer"]) + 0.5 * float(record["false_replace_rate"]) - 0.1 * float(record["improved_rate"])
    )
    return record


def selection_key(record: Mapping[str, Any], method: str) -> tuple:
    if method == "best_val":
        return (
            finite_float(record.get("wer"), 1e9),
            finite_float(record.get("cer"), 1e9),
            finite_float(record.get("worsened_rate"), 1e9),
            -finite_float(record.get("improved_rate"), 0.0),
        )
    return (
        finite_float(record.get("risk_adjusted_score"), 1e9),
        finite_float(record.get("wer"), 1e9),
        finite_float(record.get("worsened_rate"), 1e9),
        -finite_float(record.get("improved_rate"), 0.0),
    )


def rule_from_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    keys = [
        "max_rank",
        "max_edit_distance",
        "max_length_delta",
        "min_a1_delta",
        "min_mbr_delta",
        "min_teacher_delta",
        "max_function_changes",
        "max_article_changes",
        "w_mbr",
        "w_a1",
        "w_a2",
        "w_a0",
        "w_a5",
        "w_len",
        "w_edit",
    ]
    return {key: record[key] for key in keys}


def main() -> None:
    args = parse_args()
    if args.method != "rule_grid":
        raise ValueError("C2 currently supports --method rule_grid for JSON-only, no-pkl deployment.")
    output_dir = resolve_cross_platform_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(resolve_cross_platform_path(args.val_pairs_jsonl))
    groups = group_pairs(rows)
    fast_data = prepare_fast_arrays(rows)
    quick = not args.full_grid
    total_grid = grid_size(quick=quick)
    if args.max_grid_evals > 0:
        total_grid = min(total_grid, int(args.max_grid_evals))
    print(f"[C2-TRAIN] groups={len(groups)} pairs={len(rows)} grid={total_grid} mode={'quick' if quick else 'full'}")

    records: List[Dict[str, Any]] = []
    grid_path = output_dir / "grid_results.jsonl"
    with grid_path.open("w", encoding="utf-8") as handle:
        for index, rule in enumerate(grid_records(quick=quick, max_grid_evals=int(args.max_grid_evals)), start=1):
            record = evaluate_rule_fast(fast_data, rule)
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.log_every > 0 and (index == 1 or index == total_grid or index % args.log_every == 0):
                print(
                    "[C2-GRID] "
                    f"{index}/{total_grid} wer={record['wer_percent']:.4f} "
                    f"repl={record['replacements_count']} improved={record['improved_count']} "
                    f"worse={record['worsened_count']}"
                )

    if not records:
        raise ValueError("Grid produced no records")
    sorted_records = sorted(records, key=lambda record: selection_key(record, args.selection_method))
    best = sorted_records[0]
    rules = {
        "method": "rule_grid",
        "selection_method": args.selection_method,
        "selection_metric": args.selection_metric,
        "cost_false_replace": float(args.cost_false_replace),
        "cost_missed_replace": float(args.cost_missed_replace),
        "grid_mode": "quick" if quick else "full",
        "max_grid_evals": int(args.max_grid_evals),
        "rule": rule_from_record(best),
        "selected_val_metrics": best,
    }
    write_json(output_dir / "rules.json", rules)
    write_json(
        output_dir / "metrics.json",
        {
            "val_pairs_jsonl": str(resolve_cross_platform_path(args.val_pairs_jsonl)),
            "grid_results_jsonl": str(grid_path),
            "rules_json": str(output_dir / "rules.json"),
            "selected": best,
            "top20": sorted_records[:20],
        },
    )
    write_jsonl(output_dir / "top20_rules.jsonl", sorted_records[:20])
    print(json.dumps({"rules_json": str(output_dir / "rules.json"), "selected": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
