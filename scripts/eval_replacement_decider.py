"""Evaluate C2 replacement-decision rules on test pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a9_candidate_utils import char_edit_stats, prediction_reference, prediction_text, read_jsonl, word_edit_stats
from c2_replacement_utils import (
    candidate_prediction_from_pair,
    current_prediction_from_pair,
    finite_int,
    group_pairs,
    metrics_from_predictions,
    oracle_pair_for_group,
    prediction_map,
    reference_for_prediction,
    row_word_edits,
    sample_key,
    select_replacement,
    write_json,
    write_jsonl,
)
from visspeech_custom_whisper_utils import normalize_eval_text, resolve_cross_platform_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-pairs-jsonl", required=True)
    parser.add_argument("--rules-json", required=True)
    parser.add_argument("--current-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def prediction_row_from_current(raw: Mapping[str, Any], sid: str) -> Dict[str, Any]:
    ref = prediction_reference(dict(raw))
    hyp = prediction_text(dict(raw))
    word = word_edit_stats(ref, hyp)
    char = char_edit_stats(ref, hyp)
    return {
        "sample_id": sid,
        "key": sid,
        "reference": ref,
        "normalized_reference": normalize_eval_text(ref),
        "prediction": hyp,
        "normalized_prediction": normalize_eval_text(hyp),
        "selected_index": -1,
        "beam_rank": 0,
        "selector": "current_best",
        "word_edits": int(word["edits"]),
        "word_denom": int(word["denom"]),
        "sample_wer": float(word["wer"]),
        "char_edits": int(char["edits"]),
        "char_denom": int(char["denom"]),
        "sample_cer": float(char["cer"]),
    }


def oracle_prediction(group: Sequence[Mapping[str, Any]], current_row: Dict[str, Any]) -> Dict[str, Any]:
    oracle_pair = oracle_pair_for_group(group)
    if oracle_pair is None:
        return dict(current_row)
    current_edits = finite_int(current_row.get("word_edits"), 0)
    oracle_edits = finite_int(oracle_pair.get("candidate_word_edits"), 10_000)
    if oracle_edits < current_edits:
        return candidate_prediction_from_pair(oracle_pair, selector="replacement_oracle")
    return dict(current_row)


def main() -> None:
    args = parse_args()
    output_dir = resolve_cross_platform_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = read_jsonl(resolve_cross_platform_path(args.test_pairs_jsonl))
    groups = group_pairs(pairs)
    current_rows = prediction_map(read_jsonl(resolve_cross_platform_path(args.current_predictions)))
    rules_payload = json.loads(resolve_cross_platform_path(args.rules_json).read_text(encoding="utf-8"))
    rule = rules_payload.get("rule") or rules_payload

    sample_ids = sorted(set(current_rows.keys()) | set(groups.keys()))
    current_predictions: List[Dict[str, Any]] = []
    replacement_predictions: List[Dict[str, Any]] = []
    oracle_predictions: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    improved_cases: List[Dict[str, Any]] = []
    worsened_cases: List[Dict[str, Any]] = []
    missed_oracle_cases: List[Dict[str, Any]] = []

    replacements = improved = worsened = unchanged = 0
    current_correct_replaced = 0
    current_wrong_improved = 0
    current_wrong_worsened = 0

    for sid in sample_ids:
        group = groups.get(sid, [])
        if group:
            current_row = current_prediction_from_pair(group[0])
        else:
            current_row = prediction_row_from_current(current_rows.get(sid, {}), sid)
        current_predictions.append(current_row)

        selected, score = select_replacement(group, rule) if group else (None, 0.0)
        if selected is None:
            pred_row = dict(current_row)
            pred_row["selector"] = "replacement_decider_keep_current"
            decision = {
                "sample_id": sid,
                "replace": 0,
                "score": 0.0,
                "current_text": current_row.get("prediction", ""),
                "selected_text": current_row.get("prediction", ""),
                "current_word_edits": finite_int(current_row.get("word_edits"), 0),
                "selected_word_edits": finite_int(current_row.get("word_edits"), 0),
            }
        else:
            replacements += 1
            pred_row = candidate_prediction_from_pair(selected, selector="replacement_decider", score=score)
            current_edits = finite_int(selected.get("current_word_edits"), finite_int(current_row.get("word_edits"), 0))
            candidate_edits = finite_int(selected.get("candidate_word_edits"), finite_int(pred_row.get("word_edits"), 0))
            if current_edits == 0 and candidate_edits > 0:
                current_correct_replaced += 1
            if candidate_edits < current_edits:
                improved += 1
                if current_edits > 0:
                    current_wrong_improved += 1
                improved_cases.append(dict(selected))
            elif candidate_edits > current_edits:
                worsened += 1
                if current_edits > 0:
                    current_wrong_worsened += 1
                worsened_cases.append(dict(selected))
            else:
                unchanged += 1
            decision = {
                "sample_id": sid,
                "replace": 1,
                "score": float(score),
                "candidate_index": int(selected.get("candidate_index", -1)),
                "candidate_rank": int(selected.get("candidate_rank", -1)),
                "current_text": selected.get("current_text", ""),
                "selected_text": selected.get("candidate_text", ""),
                "reference": selected.get("reference", ""),
                "current_word_edits": int(current_edits),
                "selected_word_edits": int(candidate_edits),
                "delta_wer": float(selected.get("delta_wer", 0.0)),
                "mbr_z_delta": float(selected.get("mbr_z_delta", 0.0)),
                "A1_logprob_z_delta": float(selected.get("A1_logprob_z_delta", 0.0)),
                "teacher_z_delta_sum": float(selected.get("teacher_z_delta_sum", 0.0)),
            }

        replacement_predictions.append(pred_row)
        decisions.append(decision)
        oracle_row = oracle_prediction(group, current_row)
        oracle_predictions.append(oracle_row)
        if finite_int(oracle_row.get("word_edits"), 0) < finite_int(current_row.get("word_edits"), 0):
            selected_edits = finite_int(pred_row.get("word_edits"), 0)
            if selected_edits > finite_int(oracle_row.get("word_edits"), 0):
                missed_oracle_cases.append(
                    {
                        "sample_id": sid,
                        "reference": reference_for_prediction(current_row),
                        "current_prediction": current_row.get("prediction", ""),
                        "replacement_prediction": pred_row.get("prediction", ""),
                        "oracle_prediction": oracle_row.get("prediction", ""),
                        "current_word_edits": current_row.get("word_edits", 0),
                        "replacement_word_edits": pred_row.get("word_edits", 0),
                        "oracle_word_edits": oracle_row.get("word_edits", 0),
                    }
                )

    current_metrics = metrics_from_predictions(current_predictions)
    replacement_metrics = metrics_from_predictions(replacement_predictions)
    oracle_metrics = metrics_from_predictions(oracle_predictions)
    metrics = {
        "current_best": current_metrics,
        "replacement_decider": replacement_metrics,
        "oracle": oracle_metrics,
        "delta_vs_current": {
            "wer": float(replacement_metrics["wer"]) - float(current_metrics["wer"]),
            "cer": float(replacement_metrics["cer"]) - float(current_metrics["cer"]),
            "wer_percent": (float(replacement_metrics["wer"]) - float(current_metrics["wer"])) * 100.0,
            "cer_percent": (float(replacement_metrics["cer"]) - float(current_metrics["cer"])) * 100.0,
        },
        "replacements_count": replacements,
        "improved_samples": improved,
        "worsened_samples": worsened,
        "unchanged_samples": unchanged,
        "current_correct_but_replaced_count": current_correct_replaced,
        "current_wrong_and_improved_count": current_wrong_improved,
        "current_wrong_but_worsened_count": current_wrong_worsened,
        "missed_oracle_cases": len(missed_oracle_cases),
        "rules_json": str(resolve_cross_platform_path(args.rules_json)),
    }
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(output_dir / "predictions.jsonl", replacement_predictions)
    write_jsonl(output_dir / "current_predictions_aligned.jsonl", current_predictions)
    write_jsonl(output_dir / "oracle_predictions.jsonl", oracle_predictions)
    write_jsonl(output_dir / "decisions.jsonl", decisions)
    write_jsonl(output_dir / "improved_cases.jsonl", improved_cases)
    write_jsonl(output_dir / "worsened_cases.jsonl", worsened_cases)
    write_jsonl(output_dir / "missed_oracle_cases.jsonl", missed_oracle_cases)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
