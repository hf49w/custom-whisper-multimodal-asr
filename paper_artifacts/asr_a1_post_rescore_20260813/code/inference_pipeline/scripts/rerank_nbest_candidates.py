"""Untrained A9 reranking for dumped n-best candidates.

Supports ASR-only, CLIP z-score, MBR consensus score, length score, and a grid
over the linear combination:

    score = a * asr_mean_logprob + b * clip_zscore + c * mbr_score + d * length_score
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
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
    parse_float_grid,
    parse_extra_weight_specs,
    prediction_metrics,
    predictions_for_indices,
    pruned_candidate_indices,
    read_jsonl,
    resolve_cross_platform_path,
    select_by_score,
    select_by_score_from_indices,
    word_edit_stats,
    write_json,
    write_jsonl,
    write_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument(
        "--tune-jsonl",
        default="",
        help=(
            "Optional validation n-best JSONL. When set, the full parameter grid "
            "is selected on this file and the selected parameters are applied once "
            "to --input-jsonl. This prevents tuning on the test set."
        ),
    )
    parser.add_argument(
        "--dev-nbest",
        nargs="+",
        default=[],
        help=(
            "One or more development n-best JSONL files used for parameter selection. "
            "When set, these files are concatenated for grid selection and --input-jsonl "
            "is evaluated exactly once with the selected parameters."
        ),
    )
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
    parser.add_argument("--selection-method", choices=["best_val", "bootstrap_stable"], default="best_val")
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument("--bootstrap-ratio", type=float, default=1.0)
    parser.add_argument("--bootstrap-std-weight", type=float, default=0.5)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--extra-score-fields", default="", help="Comma-separated candidate fields such as A0_logprob,A1_logprob.")
    parser.add_argument(
        "--extra-score-weight-specs",
        default="",
        help=(
            "Semicolon-separated weight specs for --extra-score-fields. "
            "Example: '0,0;0.1,0.2' or 'A0_logprob=0.1,A1_logprob=0.2'."
        ),
    )
    parser.add_argument(
        "--constrained-teacher-grid",
        action="store_true",
        help=(
            "Use constrained teacher-forcing extra weights: "
            "extra_score = teacher_a0*A0_logprob + w*A1_logprob + "
            "w*A2_logprob + teacher_a5*A5_logprob."
        ),
    )
    parser.add_argument("--teacher-a0", type=float, default=0.05)
    parser.add_argument("--teacher-a5", type=float, default=0.05)
    parser.add_argument("--teacher-w-grid", default="0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument(
        "--semantic-feature-grid",
        action="store_true",
        help="Grid-search A12 semantic weights for caption_sim, tag_overlap, and visual_gain.",
    )
    parser.add_argument("--caption-sim-values", default="0,0.02,0.05,0.1,0.2")
    parser.add_argument("--tag-overlap-values", default="0,0.02,0.05,0.1,0.2")
    parser.add_argument("--visual-gain-values", default="0,0.05,0.1,0.2,0.5")
    parser.add_argument(
        "--semantic-mode",
        choices=["true", "shuffle", "disable"],
        default="true",
        help=(
            "A12 semantic ablation mode. 'shuffle' permutes semantic feature values "
            "across samples; 'disable' zeros semantic features."
        ),
    )
    parser.add_argument("--semantic-shuffle-seed", type=int, default=42)
    parser.add_argument(
        "--normalized-grid",
        action="store_true",
        help=(
            "B2 constrained normalized rerank grid. When enabled, base a/b/c/d "
            "are forced to 0 and only normalized/delta candidate fields are used."
        ),
    )
    parser.add_argument("--w-mbr-values", default="0,0.1,0.2,0.5,1.0")
    parser.add_argument("--w-a1-values", default="0,0.1,0.2,0.5,1.0")
    parser.add_argument("--w-a2-values", default="0,0.05,0.1,0.2,0.5")
    parser.add_argument("--w-a0-values", default="0,0.05,0.1,0.2")
    parser.add_argument("--w-a5-values", default="0,0.05,0.1,0.2")
    parser.add_argument("--w-len-values", default="-0.1,-0.05,0,0.05")
    parser.add_argument("--enable-pruning", action="store_true")
    parser.add_argument("--asr-top-n-values", default="5,10,15")
    parser.add_argument("--mbr-top-m-values", default="5,10")
    parser.add_argument("--clip-top-c-values", default="0,5")
    parser.add_argument("--edit-to-top1-values", default="0.2,0.3,0.5,1.0")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-scored-jsonl", action="store_true")
    return parser.parse_args()


def parse_int_grid(text: str) -> List[int]:
    return [int(float(part.strip())) for part in text.split(",") if part.strip()]


def pruning_grid(args: argparse.Namespace):
    if not args.enable_pruning:
        return [None]
    records = []
    for asr_top_n, mbr_top_m, clip_top_c, edit_to_top1 in itertools.product(
        parse_int_grid(args.asr_top_n_values),
        parse_int_grid(args.mbr_top_m_values),
        parse_int_grid(args.clip_top_c_values),
        parse_float_grid(args.edit_to_top1_values, [0.2, 0.3, 0.5, 1.0]),
    ):
        records.append(
            {
                "asr_top_n": asr_top_n,
                "mbr_top_m": mbr_top_m,
                "clip_top_c": clip_top_c,
                "edit_to_top1_max": edit_to_top1,
            }
        )
    return records


SEMANTIC_SCORE_FIELDS = [
    "caption_sim",
    "candidate_caption_similarity",
    "tag_overlap",
    "candidate_tag_overlap",
    "noun_overlap",
    "verb_overlap",
    "adjective_overlap",
    "synonym_overlap",
    "changed_word_support",
    "changed_word_penalty",
    "visual_gain",
    "supported_visual_change",
    "supported_noun_verb_change",
]


def extra_weight_grid(args: argparse.Namespace) -> List[Dict[str, float]]:
    if args.normalized_grid:
        if args.constrained_teacher_grid or args.semantic_feature_grid or args.extra_score_fields or args.extra_score_weight_specs:
            raise ValueError(
                "--normalized-grid is mutually exclusive with teacher/semantic/extra weight grids"
            )
        specs: List[Dict[str, float]] = []
        for w_mbr, w_a1, w_a2, w_a0, w_a5, w_len in itertools.product(
            parse_float_grid(args.w_mbr_values, [0.0, 0.1, 0.2, 0.5, 1.0]),
            parse_float_grid(args.w_a1_values, [0.0, 0.1, 0.2, 0.5, 1.0]),
            parse_float_grid(args.w_a2_values, [0.0, 0.05, 0.1, 0.2, 0.5]),
            parse_float_grid(args.w_a0_values, [0.0, 0.05, 0.1, 0.2]),
            parse_float_grid(args.w_a5_values, [0.0, 0.05, 0.1, 0.2]),
            parse_float_grid(args.w_len_values, [-0.1, -0.05, 0.0, 0.05]),
        ):
            specs.append(
                {
                    "mbr_z": float(w_mbr),
                    "A1_logprob_z": float(w_a1),
                    "A2_logprob_z": float(w_a2),
                    "A0_logprob_z": float(w_a0),
                    "A5_logprob_z": float(w_a5),
                    "length_delta": float(w_len),
                }
            )
        return specs

    base_specs: List[Dict[str, float]]
    if args.constrained_teacher_grid:
        if args.extra_score_fields or args.extra_score_weight_specs:
            raise ValueError("--constrained-teacher-grid cannot be combined with --extra-score-fields/--extra-score-weight-specs")
        base_specs = []
        for w in parse_float_grid(args.teacher_w_grid, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]):
            base_specs.append(
                {
                    "A0_logprob": float(args.teacher_a0),
                    "A1_logprob": float(w),
                    "A2_logprob": float(w),
                    "A5_logprob": float(args.teacher_a5),
                }
            )
    else:
        base_specs = parse_extra_weight_specs(
            args.extra_score_fields,
            args.extra_score_weight_specs,
        )
    if not args.semantic_feature_grid:
        return base_specs

    caption_values = parse_float_grid(args.caption_sim_values, [0.0, 0.02, 0.05, 0.1, 0.2])
    tag_values = parse_float_grid(args.tag_overlap_values, [0.0, 0.02, 0.05, 0.1, 0.2])
    visual_values = parse_float_grid(args.visual_gain_values, [0.0, 0.05, 0.1, 0.2, 0.5])
    specs: List[Dict[str, float]] = []
    for base, e1, e2, e3 in itertools.product(base_specs, caption_values, tag_values, visual_values):
        weights = dict(base)
        weights["caption_sim"] = float(e1)
        weights["tag_overlap"] = float(e2)
        weights["visual_gain"] = float(e3)
        specs.append(weights)
    return specs


def apply_semantic_mode(
    samples: List[Dict[str, Any]],
    *,
    mode: str,
    seed: int,
) -> List[Dict[str, Any]]:
    if mode == "true":
        return samples
    output = []
    if mode == "disable":
        for sample in samples:
            out_sample = dict(sample)
            candidates = [dict(candidate) for candidate in sample.get("candidates", [])]
            for candidate in candidates:
                for field in SEMANTIC_SCORE_FIELDS:
                    candidate[field] = 0.0
            out_sample["candidates"] = candidates
            output.append(out_sample)
        return output
    if mode != "shuffle":
        raise ValueError(f"Unsupported semantic mode: {mode}")

    rng = random.Random(int(seed))
    order = list(range(len(samples)))
    rng.shuffle(order)
    for sample_index, sample in enumerate(samples):
        source = samples[order[sample_index]]
        source_candidates = source.get("candidates", [])
        out_sample = dict(sample)
        candidates = [dict(candidate) for candidate in sample.get("candidates", [])]
        for idx, candidate in enumerate(candidates):
            source_candidate = source_candidates[min(idx, len(source_candidates) - 1)] if source_candidates else {}
            for field in SEMANTIC_SCORE_FIELDS:
                candidate[field] = float(source_candidate.get(field, 0.0))
        out_sample["candidates"] = candidates
        output.append(out_sample)
    return output


def precompute_error_tables(samples: List[Dict[str, Any]]) -> List[List[Dict[str, int]]]:
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


def bootstrap_weight_matrix(
    *,
    sample_count: int,
    iters: int,
    ratio: float,
    seed: int,
) -> np.ndarray:
    if sample_count <= 0:
        raise ValueError("Cannot bootstrap an empty sample set")
    if iters <= 0:
        raise ValueError("--bootstrap-iters must be positive for bootstrap_stable selection")
    draw_count = max(1, int(round(sample_count * float(ratio))))
    rng = np.random.default_rng(int(seed))
    weights = np.zeros((int(iters), sample_count), dtype=np.float32)
    for row in range(int(iters)):
        indices = rng.integers(0, sample_count, size=draw_count, endpoint=False)
        weights[row] = np.bincount(indices, minlength=sample_count).astype(np.float32)
    return weights


def add_bootstrap_stats_to_records(
    records: List[Dict[str, Any]],
    selections: Sequence[Dict[str, Any]],
    error_tables: List[List[Dict[str, int]]],
    *,
    iters: int,
    ratio: float,
    std_weight: float,
    seed: int,
    chunk_size: int = 256,
) -> None:
    if not records:
        return
    weights = bootstrap_weight_matrix(
        sample_count=len(error_tables),
        iters=iters,
        ratio=ratio,
        seed=seed,
    )
    for start in range(0, len(records), chunk_size):
        end = min(len(records), start + chunk_size)
        size = end - start
        word_edits = np.zeros((size, len(error_tables)), dtype=np.float32)
        word_denoms = np.ones((size, len(error_tables)), dtype=np.float32)
        char_edits = np.zeros((size, len(error_tables)), dtype=np.float32)
        char_denoms = np.ones((size, len(error_tables)), dtype=np.float32)
        for row, selection in enumerate(selections[start:end]):
            for sample_idx, raw_index in enumerate(selection["indices"]):
                table = error_tables[sample_idx]
                index = max(0, min(int(raw_index), len(table) - 1))
                stats = table[index]
                word_edits[row, sample_idx] = float(stats["word_edits"])
                word_denoms[row, sample_idx] = float(max(1, int(stats["word_denom"])))
                char_edits[row, sample_idx] = float(stats["char_edits"])
                char_denoms[row, sample_idx] = float(max(1, int(stats["char_denom"])))

        boot_word_edits = weights @ word_edits.T
        boot_word_denoms = weights @ word_denoms.T
        boot_char_edits = weights @ char_edits.T
        boot_char_denoms = weights @ char_denoms.T
        boot_wer = boot_word_edits / np.maximum(boot_word_denoms, 1.0)
        boot_cer = boot_char_edits / np.maximum(boot_char_denoms, 1.0)

        for offset, record in enumerate(records[start:end]):
            wers = boot_wer[:, offset]
            cers = boot_cer[:, offset]
            wer_mean = float(np.mean(wers))
            wer_std = float(np.std(wers))
            cer_mean = float(np.mean(cers))
            cer_std = float(np.std(cers))
            record.update(
                {
                    "bootstrap_wer_mean": wer_mean,
                    "bootstrap_wer_std": wer_std,
                    "bootstrap_wer_min": float(np.min(wers)),
                    "bootstrap_wer_max": float(np.max(wers)),
                    "bootstrap_cer_mean": cer_mean,
                    "bootstrap_cer_std": cer_std,
                    "bootstrap_cer_min": float(np.min(cers)),
                    "bootstrap_cer_max": float(np.max(cers)),
                    "bootstrap_selection_score": float(wer_mean + float(std_weight) * wer_std),
                }
            )


def raw_val_sort_key(record: Dict[str, Any]) -> tuple:
    return (float(record["wer"]), float(record["cer"]), float(record.get("mean_selected_score", 0.0)))


def bootstrap_sort_key(record: Dict[str, Any]) -> tuple:
    return (
        float(record.get("bootstrap_selection_score", record["wer"])),
        float(record.get("bootstrap_wer_mean", record["wer"])),
        float(record.get("bootstrap_wer_std", 0.0)),
        float(record["wer"]),
        float(record["cer"]),
    )


def metrics_from_indices(error_tables: List[List[Dict[str, int]]], indices: List[int]) -> Dict[str, Any]:
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


def select_indices(
    samples: List[Dict[str, Any]],
    *,
    a: float,
    b: float,
    c: float,
    d: float,
    extra_weights: Dict[str, float],
    pruning: Dict[str, Any] | None,
) -> Dict[str, Any]:
    indices: List[int] = []
    selected_scores: List[float] = []
    keep_counts: List[int] = []
    for sample in samples:
        if pruning is None:
            index, _candidate, score = select_by_score(
                sample,
                a=a,
                b=b,
                c=c,
                d=d,
                extra_weights=extra_weights,
            )
            keep_counts.append(len(sample.get("candidates", [])))
        else:
            keep = pruned_candidate_indices(sample, **pruning)
            index, _candidate, score = select_by_score_from_indices(
                sample,
                keep,
                a=a,
                b=b,
                c=c,
                d=d,
                extra_weights=extra_weights,
            )
            keep_counts.append(len(keep))
        indices.append(index)
        selected_scores.append(score)
    return {
        "indices": indices,
        "selected_scores": selected_scores,
        "keep_counts": keep_counts,
    }


def record_for_selection(
    *,
    a: float,
    b: float,
    c: float,
    d: float,
    extra_weights: Dict[str, float],
    pruning: Dict[str, Any] | None,
    error_tables: List[List[Dict[str, int]]],
    selection: Dict[str, Any],
) -> Dict[str, Any]:
    indices = selection["indices"]
    selected_scores = selection["selected_scores"]
    keep_counts = selection["keep_counts"]
    metrics = metrics_from_indices(error_tables, indices)
    return {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "extra_weights": extra_weights,
        "pruning": pruning or {},
        **metrics,
        "mean_selected_score": sum(selected_scores) / max(1, len(selected_scores)),
        "mean_kept_candidates": sum(keep_counts) / max(1, len(keep_counts)),
    }


def select_indices_from_record(samples: List[Dict[str, Any]], record: Dict[str, Any]) -> Dict[str, Any]:
    pruning = record.get("pruning") or None
    return select_indices(
        samples,
        a=float(record["a"]),
        b=float(record["b"]),
        c=float(record["c"]),
        d=float(record["d"]),
        extra_weights=dict(record.get("extra_weights") or {}),
        pruning=pruning,
    )


def run_grid(samples: List[Dict[str, Any]], args: argparse.Namespace, error_tables: List[List[Dict[str, int]]]) -> Dict[str, Any]:
    if args.normalized_grid:
        a_values = b_values = c_values = d_values = [0.0]
    else:
        a_values = parse_float_grid(args.a_values, [0.0, 0.5, 1.0])
        b_values = parse_float_grid(args.b_values, [0.0, 0.02, 0.05, 0.1])
        c_values = parse_float_grid(args.c_values, [0.0, 0.1, 0.2, 0.5])
        d_values = parse_float_grid(args.d_values, [0.0, 0.02, 0.05])
    extra_weight_specs = extra_weight_grid(args)
    pruning_specs = pruning_grid(args)
    records: List[Dict[str, Any]] = []
    selections: List[Dict[str, Any]] = []

    for a, b, c, d, extra_weights, pruning in itertools.product(
        a_values,
        b_values,
        c_values,
        d_values,
        extra_weight_specs,
        pruning_specs,
    ):
        selection = select_indices(
            samples,
            a=a,
            b=b,
            c=c,
            d=d,
            extra_weights=extra_weights,
            pruning=pruning,
        )
        record = record_for_selection(
            a=a,
            b=b,
            c=c,
            d=d,
            extra_weights=extra_weights,
            pruning=pruning,
            error_tables=error_tables,
            selection=selection,
        )
        records.append(record)
        selections.append(selection)

    if not records:
        raise RuntimeError("Empty grid")

    if args.bootstrap_iters > 0:
        add_bootstrap_stats_to_records(
            records,
            selections,
            error_tables,
            iters=int(args.bootstrap_iters),
            ratio=float(args.bootstrap_ratio),
            std_weight=float(args.bootstrap_std_weight),
            seed=int(args.bootstrap_seed),
        )
    elif args.selection_method == "bootstrap_stable":
        raise ValueError("--selection-method bootstrap_stable requires --bootstrap-iters > 0")

    raw_best_index = min(range(len(records)), key=lambda idx: raw_val_sort_key(records[idx]))
    stable_best_index = min(range(len(records)), key=lambda idx: bootstrap_sort_key(records[idx]))
    selected_best_index = stable_best_index if args.selection_method == "bootstrap_stable" else raw_best_index

    top20_raw = [records[idx] for idx in sorted(range(len(records)), key=lambda idx: raw_val_sort_key(records[idx]))[:20]]
    top20_stable = [
        records[idx]
        for idx in sorted(range(len(records)), key=lambda idx: bootstrap_sort_key(records[idx]))[:20]
    ]

    best_record = records[selected_best_index]
    best_indices = selections[selected_best_index]["indices"]
    best_predictions = predictions_for_indices(samples, best_indices, selector=f"{args.selection_method}_grid")
    return {
        "records": records,
        "best": best_record,
        "best_predictions": best_predictions,
        "best_by_raw_val": records[raw_best_index],
        "best_by_bootstrap_stable": records[stable_best_index],
        "top20_by_raw_val": top20_raw,
        "top20_by_bootstrap_stable": top20_stable,
    }


def main() -> None:
    args = parse_args()
    eval_jsonl = resolve_cross_platform_path(args.input_jsonl)
    if args.dev_nbest:
        dev_jsonl_paths = [resolve_cross_platform_path(path) for path in args.dev_nbest]
    elif args.tune_jsonl:
        dev_jsonl_paths = [resolve_cross_platform_path(args.tune_jsonl)]
    else:
        dev_jsonl_paths = [eval_jsonl]
    tune_jsonl = dev_jsonl_paths[0]
    output_dir = resolve_cross_platform_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_samples = read_jsonl(eval_jsonl)
    if not eval_samples:
        raise ValueError(f"No samples loaded from {eval_jsonl}")
    tune_uses_eval = len(dev_jsonl_paths) == 1 and str(tune_jsonl) == str(eval_jsonl)
    if tune_uses_eval:
        tune_samples = eval_samples
    else:
        tune_samples = []
        for dev_path in dev_jsonl_paths:
            rows = read_jsonl(dev_path)
            for row in rows:
                out_row = dict(row)
                out_row["_dev_source_jsonl"] = str(dev_path)
                tune_samples.append(out_row)
    if not tune_samples:
        raise ValueError(f"No tune samples loaded from {dev_jsonl_paths}")

    clip_scorer = None
    if args.clip_model_name:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        clip_scorer = ClipCandidateScorer(
            str(resolve_cross_platform_path(args.clip_model_name)),
            device=device,
            no_download=args.no_download,
        )

    eval_scored_samples = ensure_candidate_scores(eval_samples, clip_scorer=clip_scorer, log_every=args.log_every)
    tune_scored_samples = (
        eval_scored_samples
        if tune_uses_eval
        else ensure_candidate_scores(tune_samples, clip_scorer=clip_scorer, log_every=args.log_every)
    )
    eval_scored_samples = apply_semantic_mode(
        eval_scored_samples,
        mode=args.semantic_mode,
        seed=int(args.semantic_shuffle_seed),
    )
    tune_scored_samples = (
        eval_scored_samples
        if tune_uses_eval
        else apply_semantic_mode(
            tune_scored_samples,
            mode=args.semantic_mode,
            seed=int(args.semantic_shuffle_seed) + 1,
        )
    )
    if args.save_scored_jsonl:
        write_jsonl(output_dir / "scored_candidates.jsonl", eval_scored_samples)
        if not tune_uses_eval:
            write_jsonl(output_dir / "dev_scored_candidates.jsonl", tune_scored_samples)

    top1_predictions = predictions_for_indices(
        eval_scored_samples,
        [0 for _ in eval_scored_samples],
        selector="top1",
    )
    tune_top1_predictions = predictions_for_indices(
        tune_scored_samples,
        [0 for _ in tune_scored_samples],
        selector="tune_top1",
    )
    eval_curve = oracle_curve(eval_scored_samples, [1, 5, 10, 20, 30, 50])
    tune_curve = oracle_curve(tune_scored_samples, [1, 5, 10, 20, 30, 50])
    tune_error_tables = precompute_error_tables(tune_scored_samples)
    eval_error_tables = precompute_error_tables(eval_scored_samples)

    grid = run_grid(tune_scored_samples, args, tune_error_tables)
    eval_selection = select_indices_from_record(eval_scored_samples, grid["best"])
    eval_best_record = record_for_selection(
        a=float(grid["best"]["a"]),
        b=float(grid["best"]["b"]),
        c=float(grid["best"]["c"]),
        d=float(grid["best"]["d"]),
        extra_weights=dict(grid["best"].get("extra_weights") or {}),
        pruning=dict(grid["best"].get("pruning") or {}) or None,
        error_tables=eval_error_tables,
        selection=eval_selection,
    )
    eval_best_predictions = predictions_for_indices(
        eval_scored_samples,
        eval_selection["indices"],
        selector=f"{args.selection_method}_dev_tuned_grid",
    )

    write_jsonl(output_dir / "grid_metrics.jsonl", grid["records"])
    write_jsonl(output_dir / "grid_top20_raw_val.jsonl", grid["top20_by_raw_val"])
    write_jsonl(output_dir / "grid_top20_bootstrap_stable.jsonl", grid["top20_by_bootstrap_stable"])
    write_predictions(output_dir / "predictions_top1.jsonl", top1_predictions)
    write_predictions(output_dir / "predictions_best.jsonl", eval_best_predictions)
    if not tune_uses_eval:
        write_predictions(output_dir / "predictions_tune_top1.jsonl", tune_top1_predictions)
        write_predictions(output_dir / "predictions_tune_best.jsonl", grid["best_predictions"])

    summary = {
        "input_jsonl": str(eval_jsonl),
        "eval_jsonl": str(eval_jsonl),
        "tune_jsonl": str(tune_jsonl),
        "dev_nbest": [str(path) for path in dev_jsonl_paths],
        "selection_mode": "tune_on_input" if tune_uses_eval else "tune_on_dev_nbest_eval_on_input_jsonl",
        "selection_method": args.selection_method,
        "output_dir": str(output_dir),
        "rows": len(eval_scored_samples),
        "tune_rows": len(tune_scored_samples),
        "top1": prediction_metrics(top1_predictions),
        "tune_top1": prediction_metrics(tune_top1_predictions),
        "oracle_curve": eval_curve,
        "tune_oracle_curve": tune_curve,
        "best": eval_best_record,
        "tune_best": grid["best"],
        "best_by_raw_val": grid["best_by_raw_val"],
        "best_by_bootstrap_stable": grid["best_by_bootstrap_stable"],
        "top20_by_raw_val": grid["top20_by_raw_val"],
        "top20_by_bootstrap_stable": grid["top20_by_bootstrap_stable"],
        "grid_size": len(grid["records"]),
        "clip_model_name": str(resolve_cross_platform_path(args.clip_model_name)) if args.clip_model_name else "",
        "extra_score_fields": args.extra_score_fields,
        "constrained_teacher_grid": bool(args.constrained_teacher_grid),
        "teacher_a0": float(args.teacher_a0),
        "teacher_a5": float(args.teacher_a5),
        "teacher_w_grid": args.teacher_w_grid,
        "semantic_feature_grid": bool(args.semantic_feature_grid),
        "semantic_mode": args.semantic_mode,
        "semantic_shuffle_seed": int(args.semantic_shuffle_seed),
        "semantic_weight_grids": {
            "caption_sim_values": args.caption_sim_values,
            "tag_overlap_values": args.tag_overlap_values,
            "visual_gain_values": args.visual_gain_values,
        },
        "normalized_grid": bool(args.normalized_grid),
        "normalized_weight_grids": {
            "w_mbr_values": args.w_mbr_values,
            "w_a1_values": args.w_a1_values,
            "w_a2_values": args.w_a2_values,
            "w_a0_values": args.w_a0_values,
            "w_a5_values": args.w_a5_values,
            "w_len_values": args.w_len_values,
        },
        "bootstrap": {
            "iters": int(args.bootstrap_iters),
            "ratio": float(args.bootstrap_ratio),
            "std_weight": float(args.bootstrap_std_weight),
            "seed": int(args.bootstrap_seed),
        },
        "enable_pruning": bool(args.enable_pruning),
        "score_formula": (
            "normalized_grid: sum(w_field * normalized_or_delta_candidate_field); "
            "otherwise: a * asr_mean_logprob + b * clip_zscore + c * mbr_score "
            "+ d * length_score + sum(extra_weights[field] * candidate[field])"
        ),
    }
    write_json(output_dir / "metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
