"""Analyze changed cases between ASR top-1 and reranked n-best predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a9_candidate_utils import (
    ClipCandidateScorer,
    candidate_by_prediction,
    candidate_text,
    categorical_counts,
    char_edit_stats,
    edit_distance,
    ensure_candidate_scores,
    make_prediction_row,
    map_predictions,
    normalized_word_distance,
    numeric_summary,
    oracle_index,
    prediction_metrics,
    prediction_reference,
    prediction_sample_id,
    prediction_text,
    predictions_for_indices,
    read_jsonl,
    resolve_cross_platform_path,
    sample_id,
    word_edit_stats,
    write_csv,
    write_json,
    write_jsonl,
)
from visspeech_custom_whisper_utils import load_manifest, normalize_eval_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nbest-jsonl", required=True)
    parser.add_argument("--top1-predictions", required=True)
    parser.add_argument("--rerank-predictions", required=True)
    parser.add_argument("--references", default="", help="Optional manifest/jsonl with references; nbest references are used by default.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--clip-model-name", default="")
    parser.set_defaults(no_download=True)
    parser.add_argument("--no-download", dest="no_download", action="store_true")
    parser.add_argument("--allow-download", dest="no_download", action="store_false")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def load_reference_map(path_text: str) -> Dict[str, str]:
    if not path_text:
        return {}
    path = resolve_cross_platform_path(path_text)
    rows = load_manifest(path) if path.suffix.lower() in {".jsonl", ".csv"} else read_jsonl(path)
    refs: Dict[str, str] = {}
    for index, row in enumerate(rows):
        key = str(row.get("sample_id") or row.get("key") or row.get("utt_id") or Path(str(row.get("wav_path", ""))).stem or index)
        ref = str(row.get("reference") or row.get("annotation") or row.get("ref_text") or row.get("text") or "")
        refs[key] = ref
    return refs


def selected_features(
    sample: Dict[str, Any],
    top1_candidate: Dict[str, Any],
    selected_candidate: Dict[str, Any],
    *,
    selected_index: int,
    top1_text: str,
) -> Dict[str, Any]:
    asr_top1 = float(top1_candidate.get("asr_mean_logprob", top1_candidate.get("mean_logprob", 0.0)))
    asr_selected = float(selected_candidate.get("asr_mean_logprob", selected_candidate.get("mean_logprob", 0.0)))
    mbr_top1 = float(top1_candidate.get("mbr_score", 0.0))
    mbr_selected = float(selected_candidate.get("mbr_score", 0.0))
    word_count_top1 = int(top1_candidate.get("word_count", len(normalize_eval_text(top1_text).split())))
    word_count_selected = int(selected_candidate.get("word_count", len(normalize_eval_text(candidate_text(selected_candidate)).split())))
    top1_words = normalize_eval_text(top1_text).split()
    selected_words = normalize_eval_text(candidate_text(selected_candidate)).split()
    return {
        "selected_beam_rank": int(selected_candidate.get("beam_rank", selected_index)),
        "selected_index": int(selected_index),
        "asr_score_drop": asr_top1 - asr_selected,
        "mbr_gain": mbr_selected - mbr_top1,
        "clip_zscore": float(selected_candidate.get("clip_zscore", 0.0)),
        "length_difference": word_count_selected - word_count_top1,
        "abs_length_difference": abs(word_count_selected - word_count_top1),
        "edit_distance_to_top1": edit_distance(top1_words, selected_words),
        "normalized_edit_distance_to_top1": normalized_word_distance(top1_text, candidate_text(selected_candidate)),
    }


def category_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "count": len(rows),
        "selected_beam_rank": categorical_counts([row["selected_beam_rank"] for row in rows]),
        "asr_score_drop": numeric_summary([row["asr_score_drop"] for row in rows]),
        "mbr_gain": numeric_summary([row["mbr_gain"] for row in rows]),
        "clip_zscore": numeric_summary([row["clip_zscore"] for row in rows]),
        "length_difference": numeric_summary([row["length_difference"] for row in rows]),
        "abs_length_difference": numeric_summary([row["abs_length_difference"] for row in rows]),
        "edit_distance_to_top1": numeric_summary([row["edit_distance_to_top1"] for row in rows]),
        "normalized_edit_distance_to_top1": numeric_summary([row["normalized_edit_distance_to_top1"] for row in rows]),
    }


def main() -> None:
    args = parse_args()
    output_dir = resolve_cross_platform_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(resolve_cross_platform_path(args.nbest_jsonl))
    top1_map = map_predictions(read_jsonl(resolve_cross_platform_path(args.top1_predictions)))
    rerank_map = map_predictions(read_jsonl(resolve_cross_platform_path(args.rerank_predictions)))
    ref_map = load_reference_map(args.references)

    clip_scorer = None
    if args.clip_model_name:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        clip_scorer = ClipCandidateScorer(
            str(resolve_cross_platform_path(args.clip_model_name)),
            device=device,
            no_download=args.no_download,
        )
    samples = ensure_candidate_scores(samples, clip_scorer=clip_scorer, log_every=args.log_every)

    top1_predictions = []
    rerank_predictions = []
    oracle_predictions = []
    categories: Dict[str, List[Dict[str, Any]]] = {
        "top1_wrong_rerank_fixed": [],
        "top1_correct_rerank_broke": [],
        "top1_wrong_rerank_worse": [],
        "top1_wrong_oracle_fixable_rerank_missed": [],
    }
    all_cases: List[Dict[str, Any]] = []

    for index, sample in enumerate(samples):
        sid = sample_id(sample) or str(index)
        top1_pred = top1_map.get(sid)
        rerank_pred = rerank_map.get(sid)
        if top1_pred is None or rerank_pred is None:
            continue
        reference = ref_map.get(sid) or str(sample.get("reference") or prediction_reference(top1_pred) or prediction_reference(rerank_pred))
        sample["reference"] = reference
        top1_index, top1_candidate = candidate_by_prediction(sample, top1_pred)
        rerank_index, rerank_candidate = candidate_by_prediction(sample, rerank_pred)
        oracle_idx = oracle_index(sample)
        oracle_candidate = sample["candidates"][oracle_idx]

        top1_text = prediction_text(top1_pred) or candidate_text(top1_candidate)
        rerank_text = prediction_text(rerank_pred) or candidate_text(rerank_candidate)
        oracle_text = candidate_text(oracle_candidate)

        top1_word = word_edit_stats(reference, top1_text)
        rerank_word = word_edit_stats(reference, rerank_text)
        oracle_word = word_edit_stats(reference, oracle_text)
        top1_char = char_edit_stats(reference, top1_text)
        rerank_char = char_edit_stats(reference, rerank_text)
        oracle_char = char_edit_stats(reference, oracle_text)

        top1_predictions.append(make_prediction_row(sample, top1_candidate, selected_index=top1_index, selector="top1"))
        rerank_predictions.append(make_prediction_row(sample, rerank_candidate, selected_index=rerank_index, selector="rerank"))
        oracle_predictions.append(make_prediction_row(sample, oracle_candidate, selected_index=oracle_idx, selector="oracle"))

        features = selected_features(
            sample,
            top1_candidate,
            rerank_candidate,
            selected_index=rerank_index,
            top1_text=top1_text,
        )
        row = {
            "sample_id": sid,
            "reference": reference,
            "top1_prediction": top1_text,
            "rerank_prediction": rerank_text,
            "oracle_prediction": oracle_text,
            "top1_index": top1_index,
            "rerank_index": rerank_index,
            "oracle_index": oracle_idx,
            "top1_word_edits": top1_word["edits"],
            "rerank_word_edits": rerank_word["edits"],
            "oracle_word_edits": oracle_word["edits"],
            "top1_cer": top1_char["cer"],
            "rerank_cer": rerank_char["cer"],
            "oracle_cer": oracle_char["cer"],
            **features,
        }
        row_categories = []
        if top1_word["edits"] > 0 and rerank_word["edits"] == 0:
            row_categories.append("top1_wrong_rerank_fixed")
        if top1_word["edits"] == 0 and rerank_word["edits"] > 0:
            row_categories.append("top1_correct_rerank_broke")
        if top1_word["edits"] > 0 and rerank_word["edits"] > top1_word["edits"]:
            row_categories.append("top1_wrong_rerank_worse")
        if top1_word["edits"] > 0 and oracle_word["edits"] < top1_word["edits"] and rerank_word["edits"] > oracle_word["edits"]:
            row_categories.append("top1_wrong_oracle_fixable_rerank_missed")
        row["categories"] = ",".join(row_categories)
        all_cases.append(row)
        for category in row_categories:
            categories[category].append(row)

    summary = {
        "rows": len(top1_predictions),
        "top1": prediction_metrics(top1_predictions),
        "rerank": prediction_metrics(rerank_predictions),
        "oracle": prediction_metrics(oracle_predictions),
        "categories": {name: category_stats(rows) for name, rows in categories.items()},
    }
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "all_cases.jsonl", all_cases)
    write_csv(output_dir / "all_cases.csv", all_cases)
    for name, rows in categories.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
        write_csv(output_dir / f"{name}.csv", rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
