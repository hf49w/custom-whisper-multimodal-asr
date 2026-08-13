"""Compare current-best, B2 rerank, and C2 replacement-decider predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a9_candidate_utils import char_edit_stats, oracle_index, prediction_metrics, read_jsonl, word_edit_stats
from c2_replacement_utils import (
    aggregate_counts,
    changed_word_summary,
    prediction_map,
    prediction_to_reference,
    prediction_to_text,
    sample_key,
    write_json,
    write_jsonl,
)
from visspeech_custom_whisper_utils import resolve_cross_platform_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-predictions", required=True)
    parser.add_argument("--rerank-predictions", default="")
    parser.add_argument("--replacement-predictions", required=True)
    parser.add_argument("--nbest-jsonl", default="", help="Optional n-best JSONL used to derive oracle candidates.")
    parser.add_argument("--oracle-predictions", default="")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def normalize_prediction_rows(rows: List[Dict[str, Any]], selector: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        sid = sample_key(row, index)
        ref = prediction_to_reference(row)
        hyp = prediction_to_text(row)
        word = word_edit_stats(ref, hyp)
        char = char_edit_stats(ref, hyp)
        out.append(
            {
                "sample_id": sid,
                "reference": ref,
                "prediction": hyp,
                "selector": selector,
                "word_edits": int(word["edits"]),
                "word_denom": int(word["denom"]),
                "char_edits": int(char["edits"]),
                "char_denom": int(char["denom"]),
            }
        )
    return out


def aligned_row(predictions: Mapping[str, Dict[str, Any]], sid: str, fallback_ref: str = "") -> Dict[str, Any]:
    row = predictions.get(sid, {})
    ref = prediction_to_reference(row) or fallback_ref
    hyp = prediction_to_text(row)
    word = word_edit_stats(ref, hyp)
    char = char_edit_stats(ref, hyp)
    return {
        "sample_id": sid,
        "reference": ref,
        "prediction": hyp,
        "word_edits": int(word["edits"]),
        "char_edits": int(char["edits"]),
    }


def oracle_map_from_nbest(path: str) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    rows = read_jsonl(resolve_cross_platform_path(path))
    out: Dict[str, Dict[str, Any]] = {}
    for index, sample in enumerate(rows):
        sid = sample_key(sample, index)
        candidates = sample.get("candidates") or []
        if not candidates:
            continue
        oracle_idx = oracle_index(sample)
        candidate = candidates[oracle_idx]
        ref = str(sample.get("reference") or sample.get("ref_text") or "")
        hyp = str(candidate.get("text") or candidate.get("normalized_text") or "")
        word = word_edit_stats(ref, hyp)
        char = char_edit_stats(ref, hyp)
        out[sid] = {
            "sample_id": sid,
            "reference": ref,
            "prediction": hyp,
            "word_edits": int(word["edits"]),
            "char_edits": int(char["edits"]),
            "oracle_index": oracle_idx,
        }
    return out


def categories_for_change(current_text: str, replacement_text: str) -> List[str]:
    return [str(category) for category in changed_word_summary(current_text, replacement_text).get("changed_categories", [])]


def main() -> None:
    args = parse_args()
    output_dir = resolve_cross_platform_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_raw = read_jsonl(resolve_cross_platform_path(args.current_predictions))
    replacement_raw = read_jsonl(resolve_cross_platform_path(args.replacement_predictions))
    rerank_raw = read_jsonl(resolve_cross_platform_path(args.rerank_predictions)) if args.rerank_predictions else []
    oracle_raw = read_jsonl(resolve_cross_platform_path(args.oracle_predictions)) if args.oracle_predictions else []

    current = prediction_map(current_raw)
    replacement = prediction_map(replacement_raw)
    rerank = prediction_map(rerank_raw) if rerank_raw else {}
    oracle = prediction_map(oracle_raw) if oracle_raw else oracle_map_from_nbest(args.nbest_jsonl)

    sample_ids = sorted(set(current.keys()) | set(replacement.keys()) | set(rerank.keys()))
    current_aligned: List[Dict[str, Any]] = []
    replacement_aligned: List[Dict[str, Any]] = []
    rerank_aligned: List[Dict[str, Any]] = []
    oracle_aligned: List[Dict[str, Any]] = []
    replacement_better_current: List[Dict[str, Any]] = []
    replacement_worse_current: List[Dict[str, Any]] = []
    replacement_better_rerank: List[Dict[str, Any]] = []
    replacement_worse_rerank: List[Dict[str, Any]] = []
    replacement_categories: List[str] = []

    for sid in sample_ids:
        ref = prediction_to_reference(current.get(sid, {})) or prediction_to_reference(replacement.get(sid, {}))
        cur = aligned_row(current, sid, ref)
        rep = aligned_row(replacement, sid, ref)
        ren = aligned_row(rerank, sid, ref) if rerank else {}
        ora = aligned_row(oracle, sid, ref) if oracle else {}
        current_aligned.append(cur)
        replacement_aligned.append(rep)
        if ren:
            rerank_aligned.append(ren)
        if ora:
            oracle_aligned.append(ora)

        base_case = {
            "sample_id": sid,
            "reference": ref,
            "current_prediction": cur.get("prediction", ""),
            "replacement_prediction": rep.get("prediction", ""),
            "rerank_prediction": ren.get("prediction", "") if ren else "",
            "oracle_prediction": ora.get("prediction", "") if ora else "",
            "current_word_edits": cur.get("word_edits", 0),
            "replacement_word_edits": rep.get("word_edits", 0),
            "rerank_word_edits": ren.get("word_edits", 0) if ren else "",
            "oracle_word_edits": ora.get("word_edits", 0) if ora else "",
        }
        if rep["prediction"] != cur["prediction"]:
            cats = categories_for_change(cur["prediction"], rep["prediction"])
            replacement_categories.extend(cats)
            base_case["replacement_changed_categories"] = cats
        if int(rep["word_edits"]) < int(cur["word_edits"]):
            replacement_better_current.append(base_case)
        elif int(rep["word_edits"]) > int(cur["word_edits"]):
            replacement_worse_current.append(base_case)
        if ren:
            if int(rep["word_edits"]) < int(ren["word_edits"]):
                replacement_better_rerank.append(base_case)
            elif int(rep["word_edits"]) > int(ren["word_edits"]):
                replacement_worse_rerank.append(base_case)

    metrics = {
        "current_best": prediction_metrics(current_aligned),
        "replacement_decider": prediction_metrics(replacement_aligned),
        "b2_rerank": prediction_metrics(rerank_aligned) if rerank_aligned else None,
        "oracle": prediction_metrics(oracle_aligned) if oracle_aligned else None,
        "replacement_vs_current": {
            "better_samples": len(replacement_better_current),
            "worse_samples": len(replacement_worse_current),
        },
        "replacement_vs_b2_rerank": {
            "better_samples": len(replacement_better_rerank),
            "worse_samples": len(replacement_worse_rerank),
        },
        "replacement_changed_error_type_counts": aggregate_counts(replacement_categories),
    }
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(output_dir / "replacement_better_than_current.jsonl", replacement_better_current)
    write_jsonl(output_dir / "replacement_worse_than_current.jsonl", replacement_worse_current)
    write_jsonl(output_dir / "replacement_better_than_b2_rerank.jsonl", replacement_better_rerank)
    write_jsonl(output_dir / "replacement_worse_than_b2_rerank.jsonl", replacement_worse_rerank)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
