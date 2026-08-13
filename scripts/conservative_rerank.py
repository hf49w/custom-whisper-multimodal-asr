"""A10 conservative gated reranking over A9 n-best candidates."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a9_candidate_utils import (
    ClipCandidateScorer,
    char_edit_stats,
    candidate_text,
    ensure_candidate_scores,
    oracle_curve,
    parse_float_grid,
    prediction_metrics,
    predictions_for_indices,
    read_jsonl,
    resolve_cross_platform_path,
    score_candidate,
    select_by_score,
    word_edit_stats,
    write_csv,
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
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--b", type=float, default=0.01)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--d", type=float, default=0.0)
    parser.add_argument("--max-rank-values", default="2,3,5,10")
    parser.add_argument("--max-asr-drop-values", default="0.02,0.05,0.1,0.2")
    parser.add_argument("--min-mbr-gain-values", default="0,0.01,0.02,0.05")
    parser.add_argument("--min-score-margin-values", default="0,0.005,0.01,0.02,0.05")
    parser.add_argument("--max-length-delta-values", default="2,3,5,999")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-scored-jsonl", action="store_true")
    return parser.parse_args()


def parse_int_grid(text: str) -> List[int]:
    return [int(float(part.strip())) for part in text.split(",") if part.strip()]


def select_conservative(
    sample: Dict[str, Any],
    *,
    a: float,
    b: float,
    c: float,
    d: float,
    max_rank: int,
    max_asr_drop: float,
    min_mbr_gain: float,
    min_score_margin: float,
    max_length_delta: int,
) -> Tuple[int, Dict[str, Any]]:
    candidates = sample.get("candidates", [])
    if not candidates:
        raise ValueError(f"Sample has no candidates: {sample.get('sample_id')}")
    top1 = candidates[0]
    selected_index, selected, selected_score = select_by_score(sample, a=a, b=b, c=c, d=d)
    top1_score = score_candidate(top1, a=a, b=b, c=c, d=d)
    selected_human_rank = int(selected.get("beam_rank", selected_index)) + 1
    asr_drop = float(top1.get("asr_mean_logprob", top1.get("mean_logprob", 0.0))) - float(
        selected.get("asr_mean_logprob", selected.get("mean_logprob", 0.0))
    )
    mbr_gain = float(selected.get("mbr_score", 0.0)) - float(top1.get("mbr_score", 0.0))
    score_margin = float(selected_score) - float(top1_score)
    length_delta = abs(int(selected.get("word_count", 0)) - int(top1.get("word_count", 0)))
    replace = (
        selected_index != 0
        and selected_human_rank <= int(max_rank)
        and asr_drop <= float(max_asr_drop)
        and mbr_gain >= float(min_mbr_gain)
        and score_margin >= float(min_score_margin)
        and length_delta <= int(max_length_delta)
    )
    if replace:
        out = dict(selected)
        out["_gate"] = {
            "replaced": True,
            "candidate_index": selected_index,
            "candidate_human_rank": selected_human_rank,
            "asr_score_drop": asr_drop,
            "mbr_gain": mbr_gain,
            "score_margin": score_margin,
            "length_delta": length_delta,
        }
        return selected_index, out
    out = dict(top1)
    out["_gate"] = {
        "replaced": False,
        "candidate_index": selected_index,
        "candidate_human_rank": selected_human_rank,
        "asr_score_drop": asr_drop,
        "mbr_gain": mbr_gain,
        "score_margin": score_margin,
        "length_delta": length_delta,
    }
    return 0, out


def grid_values(args: argparse.Namespace):
    return itertools.product(
        parse_int_grid(args.max_rank_values),
        parse_float_grid(args.max_asr_drop_values, [0.02, 0.05, 0.1, 0.2]),
        parse_float_grid(args.min_mbr_gain_values, [0.0, 0.01, 0.02, 0.05]),
        parse_float_grid(args.min_score_margin_values, [0.0, 0.005, 0.01, 0.02, 0.05]),
        parse_int_grid(args.max_length_delta_values),
    )


def evaluate_predictions(samples: Sequence[Dict[str, Any]], indices: Sequence[int]) -> Dict[str, int]:
    top1_predictions = predictions_for_indices(samples, [0 for _ in samples], selector="top1")
    gated_predictions = predictions_for_indices(samples, indices, selector="conservative")
    improved = worse = unchanged = broke_top1 = replacements = 0
    for sample, base, pred, index in zip(samples, top1_predictions, gated_predictions, indices):
        if index != 0:
            replacements += 1
        if pred["word_edits"] < base["word_edits"]:
            improved += 1
        elif pred["word_edits"] > base["word_edits"]:
            worse += 1
            if base["word_edits"] == 0:
                broke_top1 += 1
        else:
            unchanged += 1
    return {
        "replacements": replacements,
        "improved": improved,
        "worse": worse,
        "unchanged": unchanged,
        "top1_correct_broken": broke_top1,
    }


def precompute_error_tables(samples: Sequence[Dict[str, Any]]) -> List[List[Dict[str, int]]]:
    tables: List[List[Dict[str, int]]] = []
    for sample in samples:
        reference = str(sample.get("reference", sample.get("ref_text", "")))
        row: List[Dict[str, int]] = []
        for candidate in sample.get("candidates", []):
            word = word_edit_stats(reference, candidate_text(candidate))
            char = char_edit_stats(reference, candidate_text(candidate))
            row.append(
                {
                    "word_edits": int(word["edits"]),
                    "word_denom": int(word["denom"]),
                    "char_edits": int(char["edits"]),
                    "char_denom": int(char["denom"]),
                }
            )
        if not row:
            row.append({"word_edits": 0, "word_denom": 1, "char_edits": 0, "char_denom": 1})
        tables.append(row)
    return tables


def metrics_from_indices(error_tables: Sequence[Sequence[Dict[str, int]]], indices: Sequence[int]) -> Dict[str, Any]:
    word_edits = word_denom = char_edits = char_denom = 0
    for table, raw_index in zip(error_tables, indices):
        index = max(0, min(int(raw_index), len(table) - 1))
        stats = table[index]
        word_edits += int(stats["word_edits"])
        word_denom += int(stats["word_denom"])
        char_edits += int(stats["char_edits"])
        char_denom += int(stats["char_denom"])
    return {
        "count": len(indices),
        "wer": float(word_edits / max(1, word_denom)),
        "cer": float(char_edits / max(1, char_denom)),
    }


def change_counts_from_indices(error_tables: Sequence[Sequence[Dict[str, int]]], indices: Sequence[int]) -> Dict[str, int]:
    improved = worse = unchanged = broke_top1 = replacements = 0
    for table, raw_index in zip(error_tables, indices):
        index = max(0, min(int(raw_index), len(table) - 1))
        if index != 0:
            replacements += 1
        base_edits = int(table[0]["word_edits"])
        selected_edits = int(table[index]["word_edits"])
        if selected_edits < base_edits:
            improved += 1
        elif selected_edits > base_edits:
            worse += 1
            if base_edits == 0:
                broke_top1 += 1
        else:
            unchanged += 1
    return {
        "replacements": replacements,
        "improved": improved,
        "worse": worse,
        "unchanged": unchanged,
        "top1_correct_broken": broke_top1,
    }


def main() -> None:
    args = parse_args()
    output_dir = resolve_cross_platform_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(resolve_cross_platform_path(args.input_jsonl))
    clip_scorer = None
    if args.clip_model_name:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        clip_scorer = ClipCandidateScorer(
            str(resolve_cross_platform_path(args.clip_model_name)),
            device=device,
            no_download=args.no_download,
        )
    samples = ensure_candidate_scores(samples, clip_scorer=clip_scorer, log_every=args.log_every)
    if args.save_scored_jsonl:
        write_jsonl(output_dir / "scored_candidates.jsonl", samples)

    top1_predictions = predictions_for_indices(samples, [0 for _ in samples], selector="top1")
    error_tables = precompute_error_tables(samples)
    records: List[Dict[str, Any]] = []
    best_record: Dict[str, Any] | None = None
    best_indices: List[int] = []
    best_gates: List[Dict[str, Any]] = []

    for max_rank, max_asr_drop, min_mbr_gain, min_score_margin, max_length_delta in grid_values(args):
        indices: List[int] = []
        gates: List[Dict[str, Any]] = []
        for sample in samples:
            index, selected = select_conservative(
                sample,
                a=args.a,
                b=args.b,
                c=args.c,
                d=args.d,
                max_rank=max_rank,
                max_asr_drop=max_asr_drop,
                min_mbr_gain=min_mbr_gain,
                min_score_margin=min_score_margin,
                max_length_delta=max_length_delta,
            )
            indices.append(index)
            gates.append(selected.get("_gate", {}))
        metrics = metrics_from_indices(error_tables, indices)
        changes = change_counts_from_indices(error_tables, indices)
        record = {
            "max_rank": max_rank,
            "max_asr_drop": max_asr_drop,
            "min_mbr_gain": min_mbr_gain,
            "min_score_margin": min_score_margin,
            "max_length_delta": max_length_delta,
            **metrics,
            **changes,
        }
        records.append(record)
        key = (
            record["wer"],
            record["top1_correct_broken"],
            record["worse"],
            -record["improved"],
            record["cer"],
        )
        if best_record is None or key < (
            best_record["wer"],
            best_record["top1_correct_broken"],
            best_record["worse"],
            -best_record["improved"],
            best_record["cer"],
        ):
            best_record = record
            best_indices = indices
            best_gates = gates

    if best_record is None:
        raise RuntimeError("No grid records generated")
    best_predictions = predictions_for_indices(samples, best_indices, selector="conservative_best")
    for prediction, gate in zip(best_predictions, best_gates):
        prediction.update({f"gate_{key}": value for key, value in gate.items()})
    write_jsonl(output_dir / "grid_metrics.jsonl", records)
    write_csv(output_dir / "grid_metrics.csv", records)
    write_predictions(output_dir / "predictions_top1.jsonl", top1_predictions)
    write_predictions(output_dir / "predictions_best.jsonl", best_predictions)
    changed = [prediction for prediction in best_predictions if int(prediction.get("selected_index", 0)) != 0]
    write_jsonl(output_dir / "changed_cases_best.jsonl", changed)
    write_csv(output_dir / "changed_cases_best.csv", changed)
    summary = {
        "input_jsonl": str(resolve_cross_platform_path(args.input_jsonl)),
        "rows": len(samples),
        "score_formula": "a*ASR + b*CLIP_z + c*MBR + d*length",
        "score_weights": {"a": args.a, "b": args.b, "c": args.c, "d": args.d},
        "top1": prediction_metrics(top1_predictions),
        "oracle_curve": oracle_curve(samples, [1, 5, 10, 20, 30, 50]),
        "best": best_record,
        "grid_size": len(records),
    }
    write_json(output_dir / "metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
