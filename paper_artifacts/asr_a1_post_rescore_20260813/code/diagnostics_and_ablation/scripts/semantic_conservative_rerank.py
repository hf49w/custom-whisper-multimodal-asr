"""A13 conservative semantic gated reranking for enhanced n-best candidates.

This script does not train ASR models. It selects gate thresholds on validation
or development n-best files, then evaluates the selected gate once on test.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a9_candidate_utils import (
    ClipCandidateScorer,
    candidate_text,
    char_edit_stats,
    ensure_candidate_scores,
    oracle_curve,
    parse_extra_weight_specs,
    parse_float_grid,
    prediction_metrics,
    predictions_for_indices,
    read_jsonl,
    resolve_cross_platform_path,
    score_candidate,
    select_by_score,
    word_edit_stats,
    write_json,
    write_jsonl,
    write_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--tune-jsonl", default="")
    parser.add_argument("--dev-nbest", nargs="+", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--clip-model-name", default="")
    parser.set_defaults(no_download=True)
    parser.add_argument("--no-download", dest="no_download", action="store_true")
    parser.add_argument("--allow-download", dest="no_download", action="store_false")
    parser.add_argument("--a", type=float, default=0.0)
    parser.add_argument("--b", type=float, default=0.02)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--d", type=float, default=0.1)
    parser.add_argument("--extra-score-fields", default="")
    parser.add_argument("--extra-score-weight-specs", default="")
    parser.add_argument("--constrained-teacher-grid", action="store_true")
    parser.add_argument("--teacher-a0", type=float, default=0.05)
    parser.add_argument("--teacher-a5", type=float, default=0.05)
    parser.add_argument("--teacher-w", type=float, default=0.1)
    parser.add_argument("--caption-sim-weight", type=float, default=0.0)
    parser.add_argument("--tag-overlap-weight", type=float, default=0.0)
    parser.add_argument("--visual-gain-weight", type=float, default=0.1)
    parser.add_argument("--max-rank-values", default="2,3,5,10")
    parser.add_argument("--max-asr-drop-values", default="0.02,0.05,0.1,0.2")
    parser.add_argument("--min-mbr-gain-values", default="0,0.01,0.02,0.05")
    parser.add_argument("--min-visual-gain-values", default="0,0.01,0.02,0.05")
    parser.add_argument("--require-supported-noun-verb-change", action="store_true", default=True)
    parser.add_argument("--allow-unsupported-visual-change", dest="require_supported_noun_verb_change", action="store_false")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def parse_int_grid(text: str) -> List[int]:
    return [int(float(part.strip())) for part in text.split(",") if part.strip()]


def extra_weights_from_args(args: argparse.Namespace) -> Dict[str, float]:
    if args.constrained_teacher_grid:
        if args.extra_score_fields or args.extra_score_weight_specs:
            raise ValueError("--constrained-teacher-grid cannot be combined with --extra-score-fields")
        weights = {
            "A0_logprob": float(args.teacher_a0),
            "A1_logprob": float(args.teacher_w),
            "A2_logprob": float(args.teacher_w),
            "A5_logprob": float(args.teacher_a5),
        }
    else:
        specs = parse_extra_weight_specs(args.extra_score_fields, args.extra_score_weight_specs)
        if len(specs) != 1:
            raise ValueError("A13 expects exactly one extra weight spec; grid only gates thresholds.")
        weights = dict(specs[0])
    weights["caption_sim"] = float(args.caption_sim_weight)
    weights["tag_overlap"] = float(args.tag_overlap_weight)
    weights["visual_gain"] = float(args.visual_gain_weight)
    return weights


def load_dev_samples(args: argparse.Namespace, eval_jsonl: Path) -> Tuple[List[Path], List[Dict[str, Any]], bool]:
    if args.dev_nbest:
        paths = [resolve_cross_platform_path(path) for path in args.dev_nbest]
    elif args.tune_jsonl:
        paths = [resolve_cross_platform_path(args.tune_jsonl)]
    else:
        paths = [eval_jsonl]
    tune_uses_eval = len(paths) == 1 and str(paths[0]) == str(eval_jsonl)
    if tune_uses_eval:
        return paths, [], True
    rows: List[Dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            out = dict(row)
            out["_dev_source_jsonl"] = str(path)
            rows.append(out)
    return paths, rows, False


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


def choose_a12(samples: Sequence[Dict[str, Any]], *, a: float, b: float, c: float, d: float, extra_weights: Dict[str, float]) -> List[int]:
    indices: List[int] = []
    for sample in samples:
        index, _candidate, _score = select_by_score(sample, a=a, b=b, c=c, d=d, extra_weights=extra_weights)
        indices.append(index)
    return indices


def select_conservative(
    sample: Dict[str, Any],
    *,
    a: float,
    b: float,
    c: float,
    d: float,
    extra_weights: Dict[str, float],
    max_rank: int,
    max_asr_drop: float,
    min_mbr_gain: float,
    min_visual_gain: float,
    require_supported_noun_verb_change: bool,
) -> Tuple[int, Dict[str, Any]]:
    candidates = sample.get("candidates", [])
    if not candidates:
        return 0, {"replaced": False}
    top1 = candidates[0]
    selected_index, selected, selected_score = select_by_score(sample, a=a, b=b, c=c, d=d, extra_weights=extra_weights)
    top1_score = score_candidate(top1, a=a, b=b, c=c, d=d, extra_weights=extra_weights)
    selected_rank = int(selected.get("beam_rank", selected_index)) + 1
    asr_drop = float(top1.get("asr_mean_logprob", top1.get("mean_logprob", 0.0))) - float(
        selected.get("asr_mean_logprob", selected.get("mean_logprob", 0.0))
    )
    mbr_gain = float(selected.get("mbr_score", 0.0)) - float(top1.get("mbr_score", 0.0))
    visual_gain = float(selected.get("visual_gain", 0.0))
    supported_noun_verb = bool(int(float(selected.get("supported_noun_verb_change", 0.0))))
    replace = (
        selected_index != 0
        and selected_rank <= int(max_rank)
        and asr_drop <= float(max_asr_drop)
        and mbr_gain >= float(min_mbr_gain)
        and visual_gain >= float(min_visual_gain)
        and (supported_noun_verb or not require_supported_noun_verb_change)
    )
    gate = {
        "replaced": bool(replace),
        "candidate_index": int(selected_index),
        "candidate_rank": int(selected_rank),
        "candidate_score": float(selected_score),
        "top1_score": float(top1_score),
        "score_margin": float(selected_score - top1_score),
        "asr_score_drop": float(asr_drop),
        "mbr_gain": float(mbr_gain),
        "visual_gain": float(visual_gain),
        "supported_noun_verb_change": int(supported_noun_verb),
        "changed_words": selected.get("changed_words", []),
        "supported_changed_words": selected.get("supported_changed_words", []),
    }
    return (selected_index if replace else 0), gate


def grid_values(args: argparse.Namespace):
    return itertools.product(
        parse_int_grid(args.max_rank_values),
        parse_float_grid(args.max_asr_drop_values, [0.02, 0.05, 0.1, 0.2]),
        parse_float_grid(args.min_mbr_gain_values, [0.0, 0.01, 0.02, 0.05]),
        parse_float_grid(args.min_visual_gain_values, [0.0, 0.01, 0.02, 0.05]),
    )


def change_counts(error_tables: Sequence[Sequence[Dict[str, int]]], indices: Sequence[int]) -> Dict[str, int]:
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


def tune_gates(
    samples: Sequence[Dict[str, Any]],
    error_tables: Sequence[Sequence[Dict[str, int]]],
    args: argparse.Namespace,
    extra_weights: Dict[str, float],
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    best_record: Dict[str, Any] | None = None
    best_indices: List[int] = []
    best_gates: List[Dict[str, Any]] = []
    for max_rank, max_asr_drop, min_mbr_gain, min_visual_gain in grid_values(args):
        indices: List[int] = []
        gates: List[Dict[str, Any]] = []
        for sample in samples:
            index, gate = select_conservative(
                sample,
                a=args.a,
                b=args.b,
                c=args.c,
                d=args.d,
                extra_weights=extra_weights,
                max_rank=max_rank,
                max_asr_drop=max_asr_drop,
                min_mbr_gain=min_mbr_gain,
                min_visual_gain=min_visual_gain,
                require_supported_noun_verb_change=args.require_supported_noun_verb_change,
            )
            indices.append(index)
            gates.append(gate)
        metrics = metrics_from_indices(error_tables, indices)
        record = {
            "max_rank": int(max_rank),
            "max_asr_drop": float(max_asr_drop),
            "min_mbr_gain": float(min_mbr_gain),
            "min_visual_gain": float(min_visual_gain),
            **metrics,
            **change_counts(error_tables, indices),
        }
        records.append(record)
        if best_record is None or (record["wer"], record["cer"], record["worse"]) < (
            best_record["wer"],
            best_record["cer"],
            best_record["worse"],
        ):
            best_record = record
            best_indices = indices
            best_gates = gates
    if best_record is None:
        raise RuntimeError("Empty A13 gate grid")
    return {"records": records, "best": best_record, "best_indices": best_indices, "best_gates": best_gates}


def apply_gates(samples: Sequence[Dict[str, Any]], args: argparse.Namespace, extra_weights: Dict[str, float], gate_record: Dict[str, Any]) -> Tuple[List[int], List[Dict[str, Any]]]:
    indices: List[int] = []
    gates: List[Dict[str, Any]] = []
    for sample in samples:
        index, gate = select_conservative(
            sample,
            a=args.a,
            b=args.b,
            c=args.c,
            d=args.d,
            extra_weights=extra_weights,
            max_rank=int(gate_record["max_rank"]),
            max_asr_drop=float(gate_record["max_asr_drop"]),
            min_mbr_gain=float(gate_record["min_mbr_gain"]),
            min_visual_gain=float(gate_record["min_visual_gain"]),
            require_supported_noun_verb_change=args.require_supported_noun_verb_change,
        )
        indices.append(index)
        gates.append(gate)
    return indices, gates


def case_rows(samples: Sequence[Dict[str, Any]], top1: Sequence[Dict[str, Any]], a13: Sequence[Dict[str, Any]], gates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample, base, pred, gate in zip(samples, top1, a13, gates):
        if pred["word_edits"] < base["word_edits"]:
            category = "top1_wrong_a13_improved"
        elif pred["word_edits"] > base["word_edits"]:
            category = "a13_worse"
        elif int(gate.get("replaced", 0)):
            category = "replaced_unchanged"
        else:
            category = "kept_top1"
        rows.append(
            {
                "category": category,
                "sample_id": sample.get("sample_id", sample.get("key", "")),
                "reference": base.get("reference", ""),
                "top1": base.get("prediction", ""),
                "a13": pred.get("prediction", ""),
                "top1_word_edits": base.get("word_edits"),
                "a13_word_edits": pred.get("word_edits"),
                **gate,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = resolve_cross_platform_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_jsonl = resolve_cross_platform_path(args.input_jsonl)
    eval_samples = read_jsonl(eval_jsonl)
    dev_paths, dev_samples, tune_uses_eval = load_dev_samples(args, eval_jsonl)
    if tune_uses_eval:
        dev_samples = eval_samples

    clip_scorer = None
    if args.clip_model_name:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        clip_scorer = ClipCandidateScorer(str(resolve_cross_platform_path(args.clip_model_name)), device=device, no_download=args.no_download)
    eval_samples = ensure_candidate_scores(eval_samples, clip_scorer=clip_scorer, log_every=args.log_every)
    dev_samples = eval_samples if tune_uses_eval else ensure_candidate_scores(dev_samples, clip_scorer=clip_scorer, log_every=args.log_every)

    extra_weights = extra_weights_from_args(args)
    eval_errors = precompute_error_tables(eval_samples)
    dev_errors = precompute_error_tables(dev_samples)
    top1_indices = [0 for _ in eval_samples]
    a12_indices = choose_a12(eval_samples, a=args.a, b=args.b, c=args.c, d=args.d, extra_weights=extra_weights)
    dev_a12_indices = choose_a12(dev_samples, a=args.a, b=args.b, c=args.c, d=args.d, extra_weights=extra_weights)
    tuned = tune_gates(dev_samples, dev_errors, args, extra_weights)
    a13_indices, a13_gates = apply_gates(eval_samples, args, extra_weights, tuned["best"])

    top1_predictions = predictions_for_indices(eval_samples, top1_indices, selector="top1")
    a12_predictions = predictions_for_indices(eval_samples, a12_indices, selector="a12_semantic_score")
    a13_predictions = predictions_for_indices(eval_samples, a13_indices, selector="a13_semantic_conservative")
    write_jsonl(output_dir / "grid_metrics.jsonl", tuned["records"])
    write_predictions(output_dir / "predictions_top1.jsonl", top1_predictions)
    write_predictions(output_dir / "predictions_a12.jsonl", a12_predictions)
    write_predictions(output_dir / "predictions_a13.jsonl", a13_predictions)
    write_jsonl(output_dir / "a13_case_analysis.jsonl", case_rows(eval_samples, top1_predictions, a13_predictions, a13_gates))

    summary = {
        "input_jsonl": str(eval_jsonl),
        "dev_nbest": [str(path) for path in dev_paths],
        "selection_mode": "tune_on_input" if tune_uses_eval else "tune_on_dev_nbest_eval_on_input_jsonl",
        "rows": len(eval_samples),
        "tune_rows": len(dev_samples),
        "score_weights": {"a": args.a, "b": args.b, "c": args.c, "d": args.d, "extra_weights": extra_weights},
        "top1": prediction_metrics(top1_predictions),
        "a12": prediction_metrics(a12_predictions),
        "a13": prediction_metrics(a13_predictions),
        "a12_changes": change_counts(eval_errors, a12_indices),
        "a13_changes": change_counts(eval_errors, a13_indices),
        "tune_a12": metrics_from_indices(dev_errors, dev_a12_indices),
        "tune_best_a13": tuned["best"],
        "oracle_curve": oracle_curve(eval_samples, [1, 5, 10, 20, 30, 50]),
        "require_supported_noun_verb_change": bool(args.require_supported_noun_verb_change),
    }
    write_json(output_dir / "metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
