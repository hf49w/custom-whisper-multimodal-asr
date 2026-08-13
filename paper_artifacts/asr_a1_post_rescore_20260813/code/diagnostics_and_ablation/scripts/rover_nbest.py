"""A11 word-level ROVER / confusion-network fusion for n-best candidates."""

from __future__ import annotations

import argparse
import itertools
import json
import math
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
    candidate_text,
    edit_distance,
    ensure_candidate_scores,
    oracle_curve,
    prediction_metrics,
    predictions_for_indices,
    read_jsonl,
    resolve_cross_platform_path,
    score_candidate,
    write_csv,
    write_json,
    write_jsonl,
    write_predictions,
)
from visspeech_custom_whisper_utils import normalize_eval_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k-values", default="3,5,10,20")
    parser.add_argument("--weight-modes", default="uniform,softmax_asr,softmax_asr_mbr,softmax_final_score")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--b", type=float, default=0.01)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--d", type=float, default=0.0)
    parser.add_argument("--device", default="")
    parser.add_argument("--clip-model-name", default="")
    parser.add_argument("--no-download", dest="no_download", action="store_true")
    parser.add_argument("--allow-download", dest="no_download", action="store_false")
    parser.add_argument("--log-every", type=int, default=100)
    parser.set_defaults(no_download=True)
    return parser.parse_args()


def align_ops(ref_words: Sequence[str], hyp_words: Sequence[str]) -> List[Tuple[str, int | None, int | None]]:
    n = len(ref_words)
    m = len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back: List[List[Tuple[str, int | None, int | None] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = ("delete", i - 1, None)
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = ("insert", 0, j - 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            choices = [
                (dp[i - 1][j - 1] + sub_cost, "match" if sub_cost == 0 else "sub", i - 1, j - 1),
                (dp[i - 1][j] + 1, "delete", i - 1, None),
                (dp[i][j - 1] + 1, "insert", i, j - 1),
            ]
            best = min(choices, key=lambda item: item[0])
            dp[i][j] = best[0]
            back[i][j] = (best[1], best[2], best[3])
    ops: List[Tuple[str, int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op is None:
            break
        ops.append(op)
        if op[1] is not None:
            i -= 1
        if op[2] is not None:
            j -= 1
    return list(reversed(ops))


def slot_consensus(slots: Sequence[Dict[str, float]]) -> List[str]:
    words: List[str] = []
    for slot in slots:
        non_empty = {word: weight for word, weight in slot.items() if word}
        if not non_empty:
            words.append("")
        else:
            words.append(max(non_empty.items(), key=lambda item: (item[1], item[0]))[0])
    return words


def add_candidate_to_slots(
    slots: List[Dict[str, float]],
    hyp_words: Sequence[str],
    *,
    weight: float,
    prior_weight: float,
) -> None:
    ref_words = slot_consensus(slots)
    ops = align_ops(ref_words, hyp_words)
    offset = 0
    covered = set()
    for op, ref_idx, hyp_idx in ops:
        if op == "insert":
            insert_at = (0 if ref_idx is None else ref_idx) + offset
            slot = {"": float(prior_weight)}
            slot[hyp_words[hyp_idx]] = slot.get(hyp_words[hyp_idx], 0.0) + float(weight)
            slots.insert(insert_at, slot)
            offset += 1
        elif ref_idx is not None:
            slot_idx = ref_idx + offset
            covered.add(slot_idx)
            word = "" if hyp_idx is None else hyp_words[hyp_idx]
            slots[slot_idx][word] = slots[slot_idx].get(word, 0.0) + float(weight)
    for idx, slot in enumerate(slots):
        if idx not in covered and idx >= 0:
            # Newly inserted slots were already covered by construction; adding a
            # small blank to all uncovered slots makes deletions explicit.
            slot[""] = slot.get("", 0.0) + 0.0


def final_words(slots: Sequence[Dict[str, float]]) -> List[str]:
    output: List[str] = []
    for slot in slots:
        word, _weight = max(slot.items(), key=lambda item: (item[1], item[0]))
        if word:
            output.append(word)
    return output


def softmax(values: Sequence[float], temperature: float) -> List[float]:
    if not values:
        return []
    temp = max(1e-6, float(temperature))
    scaled = [float(value) / temp for value in values]
    max_value = max(scaled)
    exps = [math.exp(value - max_value) for value in scaled]
    denom = sum(exps) or 1.0
    return [value / denom for value in exps]


def candidate_weights(
    candidates: Sequence[Dict[str, Any]],
    *,
    mode: str,
    temperature: float,
    a: float,
    b: float,
    c: float,
    d: float,
) -> List[float]:
    if mode == "uniform":
        return [1.0 for _ in candidates]
    if mode == "softmax_asr":
        values = [float(candidate.get("asr_mean_logprob", candidate.get("mean_logprob", 0.0))) for candidate in candidates]
    elif mode == "softmax_asr_mbr":
        values = [
            float(candidate.get("asr_mean_logprob", candidate.get("mean_logprob", 0.0)))
            + float(candidate.get("mbr_score", 0.0))
            for candidate in candidates
        ]
    elif mode == "softmax_final_score":
        values = [score_candidate(candidate, a=a, b=b, c=c, d=d) for candidate in candidates]
    else:
        raise ValueError(f"Unsupported weight mode: {mode}")
    return softmax(values, temperature)


def rover_sample(
    sample: Dict[str, Any],
    *,
    k: int,
    weight_mode: str,
    temperature: float,
    a: float,
    b: float,
    c: float,
    d: float,
) -> str:
    candidates = list(sample.get("candidates", []))[: max(1, int(k))]
    if not candidates:
        return ""
    weights = candidate_weights(
        candidates,
        mode=weight_mode,
        temperature=temperature,
        a=a,
        b=b,
        c=c,
        d=d,
    )
    first_words = normalize_eval_text(candidate_text(candidates[0])).split()
    slots: List[Dict[str, float]] = [{word: float(weights[0])} for word in first_words]
    total_weight = float(weights[0])
    for candidate, weight in zip(candidates[1:], weights[1:]):
        words = normalize_eval_text(candidate_text(candidate)).split()
        add_candidate_to_slots(slots, words, weight=float(weight), prior_weight=total_weight)
        total_weight += float(weight)
    return " ".join(final_words(slots)).strip()


def make_rover_predictions(samples: Sequence[Dict[str, Any]], *, k: int, weight_mode: str, args: argparse.Namespace):
    predictions = []
    for sample in samples:
        text = rover_sample(
            sample,
            k=k,
            weight_mode=weight_mode,
            temperature=args.temperature,
            a=args.a,
            b=args.b,
            c=args.c,
            d=args.d,
        )
        predictions.append(
            {
                "sample_id": sample.get("sample_id", ""),
                "key": sample.get("key", sample.get("sample_id", "")),
                "audio_path": sample.get("audio_path", sample.get("wav_path", "")),
                "image_path": sample.get("image_path", ""),
                "reference": sample.get("reference", ""),
                "prediction": text,
                "selector": f"rover_k{k}_{weight_mode}",
            }
        )
    return predictions


def main() -> None:
    args = parse_args()
    output_dir = resolve_cross_platform_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = read_jsonl(resolve_cross_platform_path(args.input_jsonl))
    clip_scorer = None
    if args.clip_model_name:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        clip_scorer = ClipCandidateScorer(
            args.clip_model_name,
            device=device,
            no_download=args.no_download,
        )
    samples = ensure_candidate_scores(samples, clip_scorer=clip_scorer, log_every=args.log_every)
    k_values = [int(part.strip()) for part in args.k_values.split(",") if part.strip()]
    weight_modes = [part.strip() for part in args.weight_modes.split(",") if part.strip()]

    top1_predictions = predictions_for_indices(samples, [0 for _ in samples], selector="top1")
    records = []
    best_record = None
    best_predictions = []
    for k, weight_mode in itertools.product(k_values, weight_modes):
        predictions = make_rover_predictions(samples, k=k, weight_mode=weight_mode, args=args)
        metrics = prediction_metrics(predictions)
        record = {"k": k, "weight_mode": weight_mode, **metrics}
        records.append(record)
        if best_record is None or (record["wer"], record["cer"]) < (best_record["wer"], best_record["cer"]):
            best_record = record
            best_predictions = predictions
        write_predictions(output_dir / f"predictions_rover_k{k}_{weight_mode}.jsonl", predictions)

    if best_record is None:
        raise RuntimeError("No ROVER configs evaluated")
    write_jsonl(output_dir / "grid_metrics.jsonl", records)
    write_csv(output_dir / "grid_metrics.csv", records)
    write_predictions(output_dir / "predictions_top1.jsonl", top1_predictions)
    write_predictions(output_dir / "predictions_best.jsonl", best_predictions)
    summary = {
        "input_jsonl": str(resolve_cross_platform_path(args.input_jsonl)),
        "rows": len(samples),
        "top1": prediction_metrics(top1_predictions),
        "oracle_curve": oracle_curve(samples, [1, 5, 10, 20, 30, 50]),
        "best": best_record,
        "grid_size": len(records),
        "temperature": args.temperature,
        "final_score_weights": {"a": args.a, "b": args.b, "c": args.c, "d": args.d},
        "clip_model_name": args.clip_model_name,
    }
    write_json(output_dir / "metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
