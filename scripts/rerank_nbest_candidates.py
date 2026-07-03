"""Untrained A9 reranking for dumped n-best candidates.

Supports ASR-only, CLIP z-score, MBR consensus score, length score, and a grid
over the linear combination:

    score = a * asr_mean_logprob + b * clip_zscore + c * mbr_score + d * length_score
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a9_candidate_utils import (
    ClipCandidateScorer,
    ensure_candidate_scores,
    oracle_curve,
    parse_float_grid,
    prediction_metrics,
    predictions_for_indices,
    read_jsonl,
    resolve_cross_platform_path,
    select_by_score,
    write_json,
    write_jsonl,
    write_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--clip-model-name", default="")
    parser.set_defaults(no_download=True)
    parser.add_argument("--no-download", dest="no_download", action="store_true")
    parser.add_argument("--allow-download", dest="no_download", action="store_false")
    parser.add_argument("--a-values", default="0,0.5,1")
    parser.add_argument("--b-values", default="0,0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--c-values", default="0,0.05,0.1,0.2,0.5,1")
    parser.add_argument("--d-values", default="0,0.01,0.02,0.05,0.1")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-scored-jsonl", action="store_true")
    return parser.parse_args()


def run_grid(samples: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    a_values = parse_float_grid(args.a_values, [0.0, 0.5, 1.0])
    b_values = parse_float_grid(args.b_values, [0.0, 0.02, 0.05, 0.1])
    c_values = parse_float_grid(args.c_values, [0.0, 0.1, 0.2, 0.5])
    d_values = parse_float_grid(args.d_values, [0.0, 0.02, 0.05])
    records: List[Dict[str, Any]] = []
    best_record: Dict[str, Any] | None = None
    best_predictions: List[Dict[str, Any]] = []

    for a, b, c, d in itertools.product(a_values, b_values, c_values, d_values):
        indices = []
        selected_scores = []
        for sample in samples:
            index, _candidate, score = select_by_score(sample, a=a, b=b, c=c, d=d)
            indices.append(index)
            selected_scores.append(score)
        predictions = predictions_for_indices(
            samples,
            indices,
            selector=f"grid_a{a:g}_b{b:g}_c{c:g}_d{d:g}",
        )
        metrics = prediction_metrics(predictions)
        record = {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            **metrics,
            "mean_selected_score": sum(selected_scores) / max(1, len(selected_scores)),
        }
        records.append(record)
        if best_record is None or (record["wer"], record["cer"]) < (best_record["wer"], best_record["cer"]):
            best_record = record
            best_predictions = predictions

    if best_record is None:
        raise RuntimeError("Empty grid")
    return {"records": records, "best": best_record, "best_predictions": best_predictions}


def main() -> None:
    args = parse_args()
    input_jsonl = resolve_cross_platform_path(args.input_jsonl)
    output_dir = resolve_cross_platform_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(input_jsonl)
    if not samples:
        raise ValueError(f"No samples loaded from {input_jsonl}")

    clip_scorer = None
    if args.clip_model_name:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        clip_scorer = ClipCandidateScorer(
            str(resolve_cross_platform_path(args.clip_model_name)),
            device=device,
            no_download=args.no_download,
        )

    scored_samples = ensure_candidate_scores(samples, clip_scorer=clip_scorer, log_every=args.log_every)
    if args.save_scored_jsonl:
        write_jsonl(output_dir / "scored_candidates.jsonl", scored_samples)

    top1_predictions = predictions_for_indices(
        scored_samples,
        [0 for _ in scored_samples],
        selector="top1",
    )
    oracle_predictions = predictions_for_indices(
        scored_samples,
        [0 if not sample.get("candidates") else min(range(len(sample["candidates"])), key=lambda idx: (
            sample["candidates"][idx].get("oracle_word_edits", 10**9),
            idx,
        )) for sample in scored_samples],
        selector="placeholder_oracle",
    )
    # Recompute oracle through the shared curve for actual top-k oracle metrics.
    curve = oracle_curve(scored_samples, [1, 5, 10, 20, 30, 50])
    grid = run_grid(scored_samples, args)

    write_jsonl(output_dir / "grid_metrics.jsonl", grid["records"])
    write_predictions(output_dir / "predictions_top1.jsonl", top1_predictions)
    write_predictions(output_dir / "predictions_best.jsonl", grid["best_predictions"])

    summary = {
        "input_jsonl": str(input_jsonl),
        "output_dir": str(output_dir),
        "rows": len(scored_samples),
        "top1": prediction_metrics(top1_predictions),
        "oracle_curve": curve,
        "best": grid["best"],
        "grid_size": len(grid["records"]),
        "clip_model_name": str(resolve_cross_platform_path(args.clip_model_name)) if args.clip_model_name else "",
        "score_formula": "a * asr_mean_logprob + b * clip_zscore + c * mbr_score + d * length_score",
    }
    write_json(output_dir / "metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
