"""Add pair-level C2 replacement-decision features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a9_candidate_utils import char_edit_stats, edit_distance, read_jsonl, word_edit_stats
from c2_replacement_utils import (
    SOURCE_NAMES,
    TEACHER_MODELS,
    changed_word_summary,
    finite_float,
    finite_int,
    write_json,
    write_jsonl,
)
from visspeech_custom_whisper_utils import normalize_eval_text, resolve_cross_platform_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--log-every", type=int, default=10000)
    return parser.parse_args()


def words(text: str) -> List[str]:
    return normalize_eval_text(text).split()


def candidate_value(row: Dict[str, Any], field: str) -> float:
    return finite_float(row.get(f"candidate_{field}"), 0.0)


def current_value(row: Dict[str, Any], field: str) -> float:
    return finite_float(row.get(f"current_{field}"), 0.0)


def add_features(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    current_text = str(out.get("current_text") or "")
    candidate_text = str(out.get("candidate_text") or "")
    current_words = words(current_text)
    candidate_words = words(candidate_text)

    word_distance = edit_distance(current_words, candidate_words)
    char_distance = edit_distance(list(normalize_eval_text(current_text)), list(normalize_eval_text(candidate_text)))
    length_delta_words = len(candidate_words) - len(current_words)
    length_delta_chars = len(normalize_eval_text(candidate_text)) - len(normalize_eval_text(current_text))

    out["candidate_rank"] = finite_int(out.get("candidate_rank"), finite_int(out.get("candidate_index"), 0) + 1)
    out["edit_distance_to_current"] = int(word_distance)
    out["word_edit_distance_to_current"] = int(word_distance)
    out["char_edit_distance_to_current"] = int(char_distance)
    out["length_delta_words"] = int(length_delta_words)
    out["length_delta_chars"] = int(length_delta_chars)
    out["abs_length_delta_words"] = abs(int(length_delta_words))
    out["abs_length_delta_chars"] = abs(int(length_delta_chars))

    out["mbr_delta"] = candidate_value(out, "mbr_score") - current_value(out, "mbr_score")
    out["mbr_z_delta"] = candidate_value(out, "mbr_z") - current_value(out, "mbr_z")
    # Keep uppercase aliases because earlier B2 files used both spellings.
    out["MBR_delta"] = out["mbr_delta"]
    out["MBR_z_delta"] = out["mbr_z_delta"]

    teacher_delta_sum = 0.0
    teacher_z_delta_sum = 0.0
    for model in TEACHER_MODELS:
        raw_delta = candidate_value(out, f"{model}_logprob") - current_value(out, f"{model}_logprob")
        z_delta = candidate_value(out, f"{model}_logprob_z") - current_value(out, f"{model}_logprob_z")
        out[f"{model}_logprob_delta"] = float(raw_delta)
        out[f"{model}_logprob_z_delta"] = float(z_delta)
        teacher_delta_sum += raw_delta
        teacher_z_delta_sum += z_delta
    out["teacher_delta_sum"] = float(teacher_delta_sum)
    out["teacher_z_delta_sum"] = float(teacher_z_delta_sum)

    for source in SOURCE_NAMES:
        out[f"from_{source}"] = finite_int(out.get(f"candidate_from_{source}"), 0)
        out[f"source_{source}_rank"] = finite_int(out.get(f"candidate_source_{source}_rank"), 10_000)
        out[f"source_{source}_logprob"] = finite_float(out.get(f"candidate_source_{source}_logprob"), 0.0)
    out["num_sources"] = finite_int(out.get("candidate_num_sources"), sum(out.get(f"from_{source}", 0) for source in SOURCE_NAMES))
    out["best_source_rank"] = finite_int(out.get("candidate_best_source_rank"), min([out[f"source_{s}_rank"] for s in SOURCE_NAMES] or [10_000]))

    changed = changed_word_summary(current_text, candidate_text)
    for key, value in changed.items():
        if key not in out or key.endswith("_count"):
            out[key] = value

    out["candidate_rank_gt_5"] = 1 if finite_int(out.get("candidate_rank"), 10_000) > 5 else 0
    out["candidate_rank_gt_10"] = 1 if finite_int(out.get("candidate_rank"), 10_000) > 10 else 0
    out["large_length_change"] = 1 if finite_int(out.get("abs_length_delta_words"), 0) > 2 else 0
    out["low_a1_score_drop"] = 1 if finite_float(out.get("A1_logprob_z_delta"), 0.0) < -0.2 else 0
    out["high_mbr_gain"] = 1 if finite_float(out.get("mbr_z_delta"), 0.0) > 0.1 else 0
    out["high_teacher_gain"] = 1 if finite_float(out.get("teacher_z_delta_sum"), 0.0) > 0.2 else 0

    # Recompute labels defensively if upstream fields are absent.
    if "current_wer" not in out or "candidate_wer" not in out:
        ref = str(out.get("reference") or "")
        current_word = word_edit_stats(ref, current_text)
        candidate_word = word_edit_stats(ref, candidate_text)
        out["current_wer"] = float(current_word["wer"])
        out["candidate_wer"] = float(candidate_word["wer"])
        out["delta_wer"] = float(current_word["wer"]) - float(candidate_word["wer"])
        out["label_replace"] = 1 if float(candidate_word["wer"]) < float(current_word["wer"]) else 0
    if "current_cer" not in out or "candidate_cer" not in out:
        ref = str(out.get("reference") or "")
        current_char = char_edit_stats(ref, current_text)
        candidate_char = char_edit_stats(ref, candidate_text)
        out["current_cer"] = float(current_char["cer"])
        out["candidate_cer"] = float(candidate_char["cer"])
        out["delta_cer"] = float(current_char["cer"]) - float(candidate_char["cer"])
    return out


def main() -> None:
    args = parse_args()
    pairs_path = resolve_cross_platform_path(args.pairs_jsonl)
    output_path = resolve_cross_platform_path(args.output_jsonl)
    summary_path = resolve_cross_platform_path(args.summary_json) if args.summary_json else output_path.with_suffix(".summary.json")
    rows = read_jsonl(pairs_path)
    out_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        out_rows.append(add_features(dict(row)))
        if args.log_every > 0 and (index == 1 or index == len(rows) or index % args.log_every == 0):
            print(f"[REPLACE-FEAT] row={index}/{len(rows)}")
    write_jsonl(output_path, out_rows)
    summary = {
        "pairs_jsonl": str(pairs_path),
        "output_jsonl": str(output_path),
        "rows": len(out_rows),
        "positive_replace_pairs": sum(1 for row in out_rows if row.get("label_replace")),
        "safe_positive_replace_pairs": sum(1 for row in out_rows if row.get("label_safe_replace")),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
