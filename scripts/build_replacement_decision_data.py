"""Build C2 replacement-decision training/eval pairs.

For each sample, the current official-best prediction is the default. Each
candidate in the union n-best list becomes a binary replacement example:
should current_best be replaced by this candidate?
"""

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

from a9_candidate_utils import (
    candidate_text,
    char_edit_stats,
    read_jsonl,
    word_edit_stats,
)
from c2_replacement_utils import (
    candidate_rank,
    candidate_raw_features,
    changed_word_summary,
    finite_float,
    prediction_map,
    prediction_to_text,
    sample_key,
    write_json,
    write_jsonl,
)
from visspeech_custom_whisper_utils import normalize_eval_text, resolve_cross_platform_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nbest-jsonl", required=True)
    parser.add_argument("--current-predictions", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-candidates-per-sample", type=int, default=50)
    parser.add_argument("--reference-field", default="reference")
    parser.add_argument(
        "--prediction-field",
        default="auto",
        help="Field to read from current predictions. Use 'auto', 'text', 'prediction', or 'normalized_text'.",
    )
    parser.add_argument("--safe-margin", type=float, default=0.0)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def reference_text(sample: Mapping[str, Any], prediction: Mapping[str, Any], field: str) -> str:
    return str(
        sample.get(field)
        or sample.get("reference")
        or sample.get("ref_text")
        or prediction.get(field)
        or prediction.get("reference")
        or prediction.get("ref_text")
        or ""
    )


def find_current_candidate(candidates: Sequence[Mapping[str, Any]], current_norm: str) -> Mapping[str, Any]:
    for candidate in candidates:
        candidate_norm = str(candidate.get("normalized_text") or normalize_eval_text(candidate_text(dict(candidate))))
        if candidate_norm == current_norm:
            return candidate
    return {}


def make_pair(
    *,
    sample: Mapping[str, Any],
    sample_id: str,
    reference: str,
    current_text: str,
    candidate: Mapping[str, Any],
    candidate_index: int,
    current_candidate: Mapping[str, Any],
    safe_margin: float,
) -> Dict[str, Any]:
    candidate_text_value = candidate_text(dict(candidate))
    current_word = word_edit_stats(reference, current_text)
    candidate_word = word_edit_stats(reference, candidate_text_value)
    current_char = char_edit_stats(reference, current_text)
    candidate_char = char_edit_stats(reference, candidate_text_value)
    current_norm = normalize_eval_text(current_text)
    candidate_norm = str(candidate.get("normalized_text") or normalize_eval_text(candidate_text_value))
    changed = changed_word_summary(current_text, candidate_text_value)
    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "key": str(sample.get("key") or sample_id),
        "audio_path": str(sample.get("audio_path") or sample.get("wav_path") or ""),
        "wav_path": str(sample.get("wav_path") or sample.get("audio_path") or ""),
        "image_path": str(sample.get("image_path") or ""),
        "reference": reference,
        "normalized_reference": normalize_eval_text(reference),
        "current_text": current_text,
        "normalized_current_text": current_norm,
        "candidate_text": candidate_text_value,
        "normalized_candidate_text": candidate_norm,
        "candidate_index": int(candidate_index),
        "candidate_rank": candidate_rank(candidate, candidate_index),
        "current_wer": float(current_word["wer"]),
        "candidate_wer": float(candidate_word["wer"]),
        "current_word_edits": int(current_word["edits"]),
        "candidate_word_edits": int(candidate_word["edits"]),
        "word_denom": int(current_word["denom"]),
        "current_cer": float(current_char["cer"]),
        "candidate_cer": float(candidate_char["cer"]),
        "current_char_edits": int(current_char["edits"]),
        "candidate_char_edits": int(candidate_char["edits"]),
        "char_denom": int(current_char["denom"]),
        "delta_wer": float(current_word["wer"]) - float(candidate_word["wer"]),
        "delta_cer": float(current_char["cer"]) - float(candidate_char["cer"]),
        "label_replace": 1 if float(candidate_word["wer"]) < float(current_word["wer"]) else 0,
        "label_safe_replace": 1
        if float(candidate_word["wer"]) + float(safe_margin) < float(current_word["wer"])
        else 0,
        "current_is_correct": 1 if int(current_word["edits"]) == 0 else 0,
        "candidate_is_correct": 1 if int(candidate_word["edits"]) == 0 else 0,
    }
    row.update(candidate_raw_features(candidate, "candidate_"))
    row.update(candidate_raw_features(current_candidate, "current_") if current_candidate else {})
    row.update(changed)
    return row


def oracle_prediction_for_sample(
    *,
    reference: str,
    current_text: str,
    candidates: Sequence[Mapping[str, Any]],
    max_candidates: int,
) -> Dict[str, Any]:
    current_word = word_edit_stats(reference, current_text)
    current_char = char_edit_stats(reference, current_text)
    best = {
        "text": current_text,
        "wer": float(current_word["wer"]),
        "cer": float(current_char["cer"]),
        "word_edits": int(current_word["edits"]),
        "char_edits": int(current_char["edits"]),
        "candidate_index": -1,
    }
    for index, candidate in enumerate(candidates[:max_candidates]):
        text = candidate_text(dict(candidate))
        word = word_edit_stats(reference, text)
        char = char_edit_stats(reference, text)
        key = (int(word["edits"]), int(char["edits"]), index)
        best_key = (int(best["word_edits"]), int(best["char_edits"]), int(best["candidate_index"]) if best["candidate_index"] >= 0 else 10_000)
        if key < best_key:
            best = {
                "text": text,
                "wer": float(word["wer"]),
                "cer": float(char["cer"]),
                "word_edits": int(word["edits"]),
                "char_edits": int(char["edits"]),
                "candidate_index": index,
            }
    return best


def main() -> None:
    args = parse_args()
    nbest_path = resolve_cross_platform_path(args.nbest_jsonl)
    current_path = resolve_cross_platform_path(args.current_predictions)
    output_path = resolve_cross_platform_path(args.output_jsonl)
    summary_path = resolve_cross_platform_path(args.summary_json) if args.summary_json else output_path.with_suffix(".summary.json")

    samples = read_jsonl(nbest_path)
    predictions = prediction_map(read_jsonl(current_path))
    rows: List[Dict[str, Any]] = []
    missing_predictions = 0
    current_correct = 0
    beneficial_samples = 0
    total_current_word_edits = total_oracle_word_edits = 0
    total_current_char_edits = total_oracle_char_edits = 0
    word_denom = char_denom = 0

    for index, sample in enumerate(samples, start=1):
        sid = sample_key(sample, index)
        prediction = predictions.get(sid, {})
        if not prediction:
            missing_predictions += 1
        reference = reference_text(sample, prediction, args.reference_field)
        current_text = prediction_to_text(prediction, args.prediction_field)
        if not current_text:
            candidates = sample.get("candidates") or []
            current_text = candidate_text(candidates[0]) if candidates else ""
        current_norm = normalize_eval_text(current_text)
        candidates = list(sample.get("candidates") or [])[: max(1, int(args.max_candidates_per_sample))]
        current_candidate = find_current_candidate(candidates, current_norm)
        current_word = word_edit_stats(reference, current_text)
        current_char = char_edit_stats(reference, current_text)
        if int(current_word["edits"]) == 0:
            current_correct += 1
        sample_has_benefit = False

        for candidate_index, candidate in enumerate(candidates):
            candidate_norm = str(candidate.get("normalized_text") or normalize_eval_text(candidate_text(dict(candidate))))
            if candidate_norm == current_norm:
                continue
            pair = make_pair(
                sample=sample,
                sample_id=sid,
                reference=reference,
                current_text=current_text,
                candidate=candidate,
                candidate_index=candidate_index,
                current_candidate=current_candidate,
                safe_margin=float(args.safe_margin),
            )
            if pair["label_replace"]:
                sample_has_benefit = True
            rows.append(pair)

        if sample_has_benefit:
            beneficial_samples += 1
        oracle = oracle_prediction_for_sample(
            reference=reference,
            current_text=current_text,
            candidates=candidates,
            max_candidates=max(1, int(args.max_candidates_per_sample)),
        )
        total_current_word_edits += int(current_word["edits"])
        total_current_char_edits += int(current_char["edits"])
        total_oracle_word_edits += int(oracle["word_edits"])
        total_oracle_char_edits += int(oracle["char_edits"])
        word_denom += int(current_word["denom"])
        char_denom += int(current_char["denom"])

        if args.log_every > 0 and (index == 1 or index == len(samples) or index % args.log_every == 0):
            print(f"[BUILD-REPLACE] sample={index}/{len(samples)} pairs={len(rows)}")

    write_jsonl(output_path, rows)
    positive = sum(1 for row in rows if row.get("label_replace"))
    safe_positive = sum(1 for row in rows if row.get("label_safe_replace"))
    summary = {
        "nbest_jsonl": str(nbest_path),
        "current_predictions": str(current_path),
        "output_jsonl": str(output_path),
        "total_samples": len(samples),
        "missing_predictions": missing_predictions,
        "total_replacement_pairs": len(rows),
        "positive_replace_pairs": positive,
        "safe_positive_replace_pairs": safe_positive,
        "samples_with_at_least_one_beneficial_replacement": beneficial_samples,
        "samples_where_current_is_already_correct": current_correct,
        "oracle_after_allowed_candidates": {
            "wer": total_oracle_word_edits / max(1, word_denom),
            "cer": total_oracle_char_edits / max(1, char_denom),
            "word_edits": total_oracle_word_edits,
            "char_edits": total_oracle_char_edits,
        },
        "current_best": {
            "wer": total_current_word_edits / max(1, word_denom),
            "cer": total_current_char_edits / max(1, char_denom),
            "word_edits": total_current_word_edits,
            "char_edits": total_current_char_edits,
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
