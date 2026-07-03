"""Evaluate a trained A9 candidate reranker on test n-best candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a9_candidate_utils import (
    ClipCandidateScorer,
    ensure_candidate_scores,
    flatten_candidate_table,
    load_pickle,
    oracle_curve,
    oracle_index,
    prediction_metrics,
    predictions_for_indices,
    read_jsonl,
    resolve_cross_platform_path,
    select_model_predictions,
    word_edit_stats,
    write_json,
    write_jsonl,
    write_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--reranker-pkl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--clip-model-name", default="")
    parser.set_defaults(no_download=True)
    parser.add_argument("--no-download", dest="no_download", action="store_true")
    parser.add_argument("--allow-download", dest="no_download", action="store_false")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-scored-jsonl", action="store_true")
    return parser.parse_args()


def model_scores(payload: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    model = payload["model"]
    scaler = payload.get("scaler")
    model_type = str(payload.get("model_type", "ridge"))
    if model_type == "ridge" and scaler is not None:
        x_used = scaler.transform(x)
    else:
        x_used = x
    return np.asarray(model.predict(x_used), dtype=np.float32)


def compare_predictions(
    top1: List[Dict[str, Any]],
    reranker: List[Dict[str, Any]],
) -> Dict[str, int]:
    improved = worse = unchanged = 0
    for base, pred in zip(top1, reranker):
        if pred["word_edits"] < base["word_edits"]:
            improved += 1
        elif pred["word_edits"] > base["word_edits"]:
            worse += 1
        else:
            unchanged += 1
    return {"improved": improved, "worse": worse, "unchanged": unchanged}


def main() -> None:
    args = parse_args()
    test_jsonl = resolve_cross_platform_path(args.test_jsonl)
    reranker_pkl = resolve_cross_platform_path(args.reranker_pkl)
    output_dir = resolve_cross_platform_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_pickle(reranker_pkl)
    samples = read_jsonl(test_jsonl)
    if not samples:
        raise ValueError(f"No samples loaded from {test_jsonl}")

    clip_model_name = args.clip_model_name or str(payload.get("clip_model_name", ""))
    clip_scorer = None
    if clip_model_name:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        clip_scorer = ClipCandidateScorer(
            str(resolve_cross_platform_path(clip_model_name)),
            device=device,
            no_download=args.no_download,
        )
    samples = ensure_candidate_scores(samples, clip_scorer=clip_scorer, log_every=args.log_every)
    if args.save_scored_jsonl:
        write_jsonl(output_dir / "test_scored_candidates.jsonl", samples)

    feature_names = payload["feature_names"]
    x, _y, _groups, index_pairs, _sample_ids = flatten_candidate_table(
        samples,
        feature_names=feature_names,
        with_labels=False,
    )
    scores = model_scores(payload, x)
    reranker_predictions = select_model_predictions(
        samples,
        scores,
        index_pairs,
        selector="reranker",
    )

    top1_predictions = predictions_for_indices(
        samples,
        [0 for _ in samples],
        selector="top1",
    )
    oracle_indices = [oracle_index(sample, limit=None) for sample in samples]
    oracle_predictions = predictions_for_indices(
        samples,
        oracle_indices,
        selector="oracle_all",
    )
    curve = oracle_curve(samples, [1, 5, 10, 20, 30, 50])
    comparison = compare_predictions(top1_predictions, reranker_predictions)

    oracle_missed: List[Dict[str, Any]] = []
    error_cases: List[Dict[str, Any]] = []
    for sample, top1, oracle, reranked in zip(
        samples,
        top1_predictions,
        oracle_predictions,
        reranker_predictions,
    ):
        oracle_can_fix = oracle["word_edits"] < top1["word_edits"]
        reranker_missed = reranked["word_edits"] > oracle["word_edits"]
        if oracle_can_fix and reranker_missed:
            oracle_missed.append(
                {
                    "sample_id": sample.get("sample_id", ""),
                    "reference": sample.get("reference", ""),
                    "top1_prediction": top1["prediction"],
                    "top1_edits": top1["word_edits"],
                    "oracle_prediction": oracle["prediction"],
                    "oracle_index": oracle["selected_index"],
                    "oracle_edits": oracle["word_edits"],
                    "reranker_prediction": reranked["prediction"],
                    "reranker_index": reranked["selected_index"],
                    "reranker_edits": reranked["word_edits"],
                    "candidates": sample.get("candidates", []),
                }
            )
        if reranked["word_edits"] > 0:
            error_cases.append(
                {
                    "sample_id": sample.get("sample_id", ""),
                    "reference": sample.get("reference", ""),
                    "top1_prediction": top1["prediction"],
                    "oracle_prediction": oracle["prediction"],
                    "reranker_prediction": reranked["prediction"],
                    "top1_edits": top1["word_edits"],
                    "oracle_edits": oracle["word_edits"],
                    "reranker_edits": reranked["word_edits"],
                    "reranker_index": reranked["selected_index"],
                }
            )

    write_predictions(output_dir / "predictions_top1.jsonl", top1_predictions)
    write_predictions(output_dir / "predictions_oracle.jsonl", oracle_predictions)
    write_predictions(output_dir / "predictions_reranker.jsonl", reranker_predictions)
    write_jsonl(output_dir / "oracle_missed.jsonl", oracle_missed)
    write_jsonl(output_dir / "error_cases.jsonl", error_cases)

    summary = {
        "test_jsonl": str(test_jsonl),
        "reranker_pkl": str(reranker_pkl),
        "rows": len(samples),
        "top1": prediction_metrics(top1_predictions),
        "oracle": prediction_metrics(oracle_predictions),
        "oracle_curve": curve,
        "reranker": prediction_metrics(reranker_predictions),
        "comparison_vs_top1": comparison,
        "oracle_missed_count": len(oracle_missed),
        "error_case_count": len(error_cases),
        "feature_names": feature_names,
        "model_type": payload.get("model_type", ""),
        "params": payload.get("params", {}),
        "clip_model_name": str(resolve_cross_platform_path(clip_model_name)) if clip_model_name else "",
    }
    write_json(output_dir / "metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
