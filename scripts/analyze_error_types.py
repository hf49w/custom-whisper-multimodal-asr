"""Analyze ASR error types and oracle-fixable n-best cases."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a9_candidate_utils import (
    candidate_text,
    edit_distance,
    map_predictions,
    normalize_eval_text,
    oracle_index,
    prediction_sample_id,
    prediction_text,
    read_jsonl,
    word_edit_stats,
    char_edit_stats,
)
from visspeech_custom_whisper_utils import resolve_cross_platform_path


ARTICLES = {"a", "an", "the"}
PREPOSITIONS = {
    "about", "above", "across", "after", "against", "along", "around", "at", "behind",
    "below", "beside", "between", "by", "down", "for", "from", "in", "inside", "into",
    "near", "of", "off", "on", "onto", "over", "through", "to", "under", "up", "with",
}
FUNCTION_WORDS = ARTICLES | PREPOSITIONS | {
    "and", "or", "but", "is", "are", "was", "were", "be", "been", "being", "it", "its",
    "he", "she", "they", "them", "his", "her", "their", "this", "that", "these", "those",
}
COLORS = {"black", "blue", "brown", "gray", "green", "grey", "orange", "pink", "purple", "red", "white", "yellow"}
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "first", "second", "third", "single", "double", "couple", "many", "several",
}
COMMON_VERBS = {
    "bike", "biking", "carry", "carrying", "climb", "climbing", "dance", "dancing",
    "drive", "driving", "eat", "eating", "hold", "holding", "jump", "jumping",
    "look", "looking", "play", "playing", "ride", "riding", "run", "running",
    "sit", "sitting", "skate", "skating", "ski", "skiing", "slide", "sliding",
    "smile", "smiling", "snowboard", "snowboarding", "stand", "standing", "swim",
    "swimming", "throw", "throwing", "walk", "walking", "wear", "wearing",
}
COMMON_ADJECTIVES = {
    "big", "black", "blue", "brown", "dirty", "green", "large", "little", "old",
    "orange", "red", "small", "tall", "white", "yellow", "young",
}
VISUAL_HELPFUL = {"noun", "verb", "adjective", "number", "color"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nbest-jsonl", required=True)
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--oracle-predictions-jsonl", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--oracle-limit", type=int, default=0)
    return parser.parse_args()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def words(text: str) -> List[str]:
    return normalize_eval_text(text).split()


def classify_word(word: str) -> str:
    token = normalize_eval_text(word).strip()
    if not token:
        return "empty"
    if token in ARTICLES:
        return "article"
    if token in PREPOSITIONS:
        return "preposition"
    if token in COLORS:
        return "color"
    if token in NUMBER_WORDS or token.isdigit():
        return "number"
    if token in COMMON_VERBS or token.endswith("ing") or token.endswith("ed"):
        return "verb"
    if token in COMMON_ADJECTIVES:
        return "adjective"
    if token in FUNCTION_WORDS:
        return "plural_function"
    if token.endswith("s") and len(token) > 3:
        return "plural_function"
    return "noun"


def edit_ops(ref_words: Sequence[str], hyp_words: Sequence[str]) -> List[Dict[str, Any]]:
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    back = [[""] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
        back[i][0] = "delete"
    for j in range(1, n + 1):
        dp[0][j] = j
        back[0][j] = "insert"
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "equal"
            else:
                choices = [
                    (dp[i - 1][j - 1] + 1, "replace"),
                    (dp[i][j - 1] + 1, "insert"),
                    (dp[i - 1][j] + 1, "delete"),
                ]
                dp[i][j], back[i][j] = min(choices, key=lambda item: item[0])
    ops: List[Dict[str, Any]] = []
    i, j = m, n
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "equal":
            i -= 1
            j -= 1
            continue
        if op == "replace":
            ref_word = ref_words[i - 1]
            hyp_word = hyp_words[j - 1]
            i -= 1
            j -= 1
        elif op == "insert":
            ref_word = ""
            hyp_word = hyp_words[j - 1]
            j -= 1
        else:
            ref_word = ref_words[i - 1]
            hyp_word = ""
            i -= 1
        focus = hyp_word or ref_word
        ops.append(
            {
                "op": op,
                "ref_word": ref_word,
                "hyp_word": hyp_word,
                "category": classify_word(focus),
            }
        )
    return list(reversed(ops))


def counts(items: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def sample_id(sample: Dict[str, Any], index: int) -> str:
    return str(sample.get("sample_id") or sample.get("key") or sample.get("utt_id") or index)


def oracle_text(sample: Dict[str, Any], oracle_predictions: Dict[str, Dict[str, Any]], key: str, limit: int) -> Tuple[str, int]:
    if oracle_predictions and key in oracle_predictions:
        pred = oracle_predictions[key]
        return prediction_text(pred), int(pred.get("selected_index", pred.get("beam_rank", 0)) or 0)
    index = oracle_index(sample, limit=limit if limit > 0 else None)
    candidates = sample.get("candidates", [])
    if not candidates:
        return "", 0
    return candidate_text(candidates[index]), int(index)


def main() -> None:
    args = parse_args()
    output_dir = resolve_cross_platform_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = read_jsonl(resolve_cross_platform_path(args.nbest_jsonl))
    predictions = map_predictions(read_jsonl(resolve_cross_platform_path(args.predictions_jsonl)))
    oracle_predictions = (
        map_predictions(read_jsonl(resolve_cross_platform_path(args.oracle_predictions_jsonl)))
        if args.oracle_predictions_jsonl
        else {}
    )

    rows: List[Dict[str, Any]] = []
    current_categories: List[str] = []
    oracle_fixable_categories: List[str] = []
    visual_helpful_fixable = 0
    oracle_better = 0
    total_current_word_edits = total_oracle_word_edits = 0
    total_current_char_edits = total_oracle_char_edits = 0
    word_denom = char_denom = 0
    for index, sample in enumerate(samples, start=1):
        key = sample_id(sample, index)
        prediction = predictions.get(key) or {}
        ref = str(sample.get("reference") or sample.get("ref_text") or "")
        current_text = prediction_text(prediction) or candidate_text((sample.get("candidates") or [{}])[0])
        oracle_hyp, oracle_idx = oracle_text(sample, oracle_predictions, key, int(args.oracle_limit))
        current_word = word_edit_stats(ref, current_text)
        oracle_word = word_edit_stats(ref, oracle_hyp)
        current_char = char_edit_stats(ref, current_text)
        oracle_char = char_edit_stats(ref, oracle_hyp)
        total_current_word_edits += int(current_word["edits"])
        total_oracle_word_edits += int(oracle_word["edits"])
        total_current_char_edits += int(current_char["edits"])
        total_oracle_char_edits += int(oracle_char["edits"])
        word_denom += int(current_word["denom"])
        char_denom += int(current_char["denom"])

        ops = edit_ops(words(ref), words(current_text))
        oracle_ops = edit_ops(words(ref), words(oracle_hyp))
        cats = [op["category"] for op in ops]
        current_categories.extend(cats)
        is_oracle_better = int(oracle_word["edits"]) < int(current_word["edits"])
        if is_oracle_better:
            oracle_better += 1
            oracle_fixable_categories.extend(cats)
            if any(category in VISUAL_HELPFUL for category in cats):
                visual_helpful_fixable += 1
        rows.append(
            {
                "sample_id": key,
                "reference": ref,
                "current_prediction": current_text,
                "oracle_prediction": oracle_hyp,
                "oracle_index": oracle_idx,
                "current_word_edits": int(current_word["edits"]),
                "oracle_word_edits": int(oracle_word["edits"]),
                "current_cer_edits": int(current_char["edits"]),
                "oracle_cer_edits": int(oracle_char["edits"]),
                "oracle_better": bool(is_oracle_better),
                "visual_possible_helpful": bool(any(category in VISUAL_HELPFUL for category in cats)),
                "error_categories": counts(cats),
                "oracle_error_categories": counts([op["category"] for op in oracle_ops]),
                "edit_operations": ops,
            }
        )

    summary = {
        "nbest_jsonl": str(resolve_cross_platform_path(args.nbest_jsonl)),
        "predictions_jsonl": str(resolve_cross_platform_path(args.predictions_jsonl)),
        "oracle_predictions_jsonl": str(resolve_cross_platform_path(args.oracle_predictions_jsonl)) if args.oracle_predictions_jsonl else "",
        "rows": len(samples),
        "current_wer": total_current_word_edits / max(1, word_denom),
        "oracle_wer": total_oracle_word_edits / max(1, word_denom),
        "current_cer": total_current_char_edits / max(1, char_denom),
        "oracle_cer": total_oracle_char_edits / max(1, char_denom),
        "oracle_better_samples": oracle_better,
        "visual_possible_helpful_oracle_fixable_samples": visual_helpful_fixable,
        "visual_possible_helpful_oracle_fixable_ratio": visual_helpful_fixable / max(1, oracle_better),
        "current_error_type_counts": counts(current_categories),
        "oracle_fixable_error_type_counts": counts(oracle_fixable_categories),
        "visual_helpful_categories": sorted(VISUAL_HELPFUL),
    }
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "cases.jsonl", rows)
    flat_rows = []
    for row in rows:
        flat = dict(row)
        flat["error_categories"] = json.dumps(flat["error_categories"], ensure_ascii=False)
        flat["oracle_error_categories"] = json.dumps(flat["oracle_error_categories"], ensure_ascii=False)
        flat["edit_operations"] = json.dumps(flat["edit_operations"], ensure_ascii=False)
        flat_rows.append(flat)
    write_csv(output_dir / "cases.csv", flat_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
