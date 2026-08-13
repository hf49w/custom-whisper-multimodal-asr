"""Shared utilities for C2 replacement-decision reranking.

C2 treats reranking as a conservative replacement decision: keep the current
best prediction unless a candidate passes risk gates and has the highest
decision score under validation-selected rules.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from a9_candidate_utils import (
    char_edit_stats,
    edit_distance,
    prediction_metrics,
    prediction_reference,
    prediction_sample_id,
    prediction_text,
    word_edit_stats,
)
from visspeech_custom_whisper_utils import normalize_eval_text


ARTICLES = {"a", "an", "the"}
PREPOSITIONS = {
    "in",
    "on",
    "at",
    "with",
    "by",
    "for",
    "from",
    "to",
    "into",
    "over",
    "under",
    "of",
    "off",
    "onto",
    "near",
    "behind",
    "beside",
    "around",
    "through",
    "across",
}
COLORS = {"red", "blue", "green", "yellow", "black", "white", "brown", "orange", "pink", "purple", "gray", "grey"}
NUMBERS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "first",
    "second",
    "third",
    "single",
    "double",
    "couple",
}
FUNCTION_WORDS = ARTICLES | PREPOSITIONS | {
    "and",
    "or",
    "but",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "he",
    "she",
    "they",
    "them",
    "his",
    "her",
    "their",
    "this",
    "that",
    "these",
    "those",
}
COMMON_VERBS = {
    "bike",
    "biking",
    "carry",
    "carrying",
    "climb",
    "climbing",
    "dance",
    "dancing",
    "drive",
    "driving",
    "eat",
    "eating",
    "hold",
    "holding",
    "jump",
    "jumping",
    "look",
    "looking",
    "play",
    "playing",
    "ride",
    "riding",
    "run",
    "running",
    "sit",
    "sitting",
    "skate",
    "skating",
    "ski",
    "skiing",
    "slide",
    "sliding",
    "smile",
    "smiling",
    "snowboard",
    "snowboarding",
    "stand",
    "standing",
    "swim",
    "swimming",
    "throw",
    "throwing",
    "walk",
    "walking",
    "wear",
    "wearing",
}
COMMON_ADJECTIVES = {
    "big",
    "dirty",
    "large",
    "little",
    "old",
    "small",
    "tall",
    "young",
    "new",
    "short",
    "long",
    "wooden",
    "grassy",
    "snowy",
} | COLORS

SOURCE_NAMES = ["A0", "A1b10", "A1b20", "A1b30", "A2", "A5"]
TEACHER_MODELS = ["A1", "A2", "A0", "A5"]


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def finite_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_key(row: Mapping[str, Any], fallback: int | str = "") -> str:
    return str(row.get("sample_id") or row.get("key") or row.get("utt_id") or fallback)


def prediction_to_text(row: Mapping[str, Any], field: str = "auto") -> str:
    if field and field != "auto" and row.get(field) is not None:
        return str(row.get(field) or "")
    return prediction_text(dict(row))


def prediction_to_reference(row: Mapping[str, Any]) -> str:
    return prediction_reference(dict(row))


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
    if token in NUMBERS or token.isdigit():
        return "number"
    if token in COMMON_VERBS or token.endswith("ing") or token.endswith("ed"):
        return "verb"
    if token in COMMON_ADJECTIVES:
        return "adjective"
    if token in FUNCTION_WORDS:
        return "function"
    if token.endswith("s") and len(token) > 3:
        return "plural_like"
    return "noun"


def edit_ops(source_words: Sequence[str], target_words: Sequence[str]) -> List[Dict[str, Any]]:
    """Return word edit ops from source/current to target/candidate."""

    m, n = len(source_words), len(target_words)
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
            if source_words[i - 1] == target_words[j - 1]:
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
            source_word = source_words[i - 1]
            target_word = target_words[j - 1]
            i -= 1
            j -= 1
        elif op == "insert":
            source_word = ""
            target_word = target_words[j - 1]
            j -= 1
        else:
            source_word = source_words[i - 1]
            target_word = ""
            i -= 1
        focus = target_word or source_word
        ops.append(
            {
                "op": op,
                "current_word": source_word,
                "candidate_word": target_word,
                "word": focus,
                "category": classify_word(focus),
            }
        )
    return list(reversed(ops))


def changed_word_summary(current_text: str, candidate_text: str) -> Dict[str, Any]:
    ops = edit_ops(words(current_text), words(candidate_text))
    categories = [str(op["category"]) for op in ops]
    counts: Dict[str, int] = {}
    for category in categories:
        counts[category] = counts.get(category, 0) + 1
    return {
        "changed_words": [op["word"] for op in ops],
        "changed_ops": ops,
        "changed_categories": categories,
        "changed_noun_count": counts.get("noun", 0),
        "changed_verb_count": counts.get("verb", 0),
        "changed_adj_count": counts.get("adjective", 0),
        "changed_color_count": counts.get("color", 0),
        "changed_number_count": counts.get("number", 0),
        "changed_article_count": counts.get("article", 0),
        "changed_preposition_count": counts.get("preposition", 0),
        "changed_function_count": counts.get("function", 0),
        "changed_plural_like_count": counts.get("plural_like", 0),
    }


def source_present(candidate: Mapping[str, Any], source: str) -> int:
    if source in set(candidate.get("sources") or []):
        return 1
    return finite_int(candidate.get(f"source_{source}_present"), 0)


def source_rank(candidate: Mapping[str, Any], source: str) -> int:
    source_scores = candidate.get("source_scores") or {}
    if isinstance(source_scores, Mapping) and source in source_scores:
        return finite_int((source_scores.get(source) or {}).get("beam_rank"), 10_000)
    return finite_int(candidate.get(f"source_{source}_rank"), 10_000)


def source_logprob(candidate: Mapping[str, Any], source: str) -> float:
    source_scores = candidate.get("source_scores") or {}
    if isinstance(source_scores, Mapping) and source in source_scores:
        item = source_scores.get(source) or {}
        return finite_float(item.get("mean_logprob", item.get("asr_mean_logprob", 0.0)))
    return finite_float(candidate.get(f"source_{source}_logprob"), 0.0)


def best_source_rank(candidate: Mapping[str, Any]) -> int:
    ranks = [source_rank(candidate, source) for source in SOURCE_NAMES if source_present(candidate, source)]
    rank = min(ranks) if ranks else finite_int(candidate.get("source_min_rank", candidate.get("beam_rank", 10_000)), 10_000)
    return int(rank)


def candidate_rank(candidate: Mapping[str, Any], candidate_index: int) -> int:
    # Use the union-list rank as the replacement gate rank. It is stable across
    # sources and matches the candidate pool being searched.
    return int(candidate_index) + 1


def candidate_raw_features(candidate: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    fields = [
        "beam_rank",
        "source_min_rank",
        "source_count",
        "source_best_logprob",
        "mean_logprob",
        "asr_mean_logprob",
        "sum_logprob",
        "avg_token_logprob",
        "min_token_logprob",
        "token_count",
        "word_count",
        "char_count",
        "compression_ratio",
        "mbr",
        "mbr_score",
        "mbr_z",
        "MBR_z",
        "mbr_delta",
        "MBR_delta",
        "length_delta",
        "length_abs_delta",
    ]
    for model in TEACHER_MODELS:
        fields.extend(
            [
                f"{model}_logprob",
                f"{model}_sum_logprob",
                f"{model}_min_token_logprob",
                f"{model}_logprob_z",
                f"{model}_logprob_delta",
            ]
        )
    out: Dict[str, Any] = {}
    for field in fields:
        value = candidate.get(field)
        if isinstance(value, (int, float)) or value is None:
            out[f"{prefix}{field}"] = finite_float(value, 0.0)
    for source in SOURCE_NAMES:
        out[f"{prefix}from_{source}"] = source_present(candidate, source)
        out[f"{prefix}source_{source}_rank"] = source_rank(candidate, source)
        out[f"{prefix}source_{source}_logprob"] = source_logprob(candidate, source)
    out[f"{prefix}num_sources"] = finite_int(candidate.get("source_count"), len(candidate.get("sources") or []))
    out[f"{prefix}best_source_rank"] = best_source_rank(candidate)
    return out


def pair_prediction_row(pair: Mapping[str, Any], *, use_candidate: bool, selector: str, score: float = 0.0) -> Dict[str, Any]:
    ref = str(pair.get("reference") or "")
    hyp = str(pair.get("candidate_text") if use_candidate else pair.get("current_text") or "")
    word = word_edit_stats(ref, hyp)
    char = char_edit_stats(ref, hyp)
    return {
        "sample_id": str(pair.get("sample_id") or ""),
        "key": str(pair.get("sample_id") or ""),
        "audio_path": str(pair.get("audio_path") or ""),
        "wav_path": str(pair.get("wav_path") or pair.get("audio_path") or ""),
        "image_path": str(pair.get("image_path") or ""),
        "reference": ref,
        "normalized_reference": normalize_eval_text(ref),
        "prediction": hyp,
        "normalized_prediction": normalize_eval_text(hyp),
        "selected_index": int(pair.get("candidate_index", 0) if use_candidate else -1),
        "beam_rank": int(pair.get("candidate_rank", 0) if use_candidate else 0),
        "selector": selector,
        "replacement_score": float(score),
        "word_edits": int(word["edits"]),
        "word_denom": int(word["denom"]),
        "sample_wer": float(word["wer"]),
        "char_edits": int(char["edits"]),
        "char_denom": int(char["denom"]),
        "sample_cer": float(char["cer"]),
    }


def group_pairs(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("sample_id") or "")
        groups.setdefault(key, []).append(dict(row))
    return groups


def teacher_delta_sum(pair: Mapping[str, Any]) -> float:
    return sum(finite_float(pair.get(f"{model}_logprob_z_delta"), 0.0) for model in TEACHER_MODELS)


def rule_passes(pair: Mapping[str, Any], rule: Mapping[str, Any]) -> bool:
    return (
        finite_float(pair.get("candidate_rank"), 10_000) <= finite_float(rule.get("max_rank"), 0)
        and finite_float(pair.get("edit_distance_to_current"), 10_000) <= finite_float(rule.get("max_edit_distance"), 0)
        and finite_float(pair.get("abs_length_delta_words"), 10_000) <= finite_float(rule.get("max_length_delta"), 0)
        and finite_float(pair.get("A1_logprob_z_delta"), 0.0) >= finite_float(rule.get("min_a1_delta"), -math.inf)
        and finite_float(pair.get("mbr_z_delta"), 0.0) >= finite_float(rule.get("min_mbr_delta"), -math.inf)
        and teacher_delta_sum(pair) >= finite_float(rule.get("min_teacher_delta"), -math.inf)
        and finite_float(pair.get("changed_function_count"), 0.0) <= finite_float(rule.get("max_function_changes"), math.inf)
        and finite_float(pair.get("changed_article_count"), 0.0) <= finite_float(rule.get("max_article_changes"), math.inf)
    )


def decision_score(pair: Mapping[str, Any], rule: Mapping[str, Any]) -> float:
    return (
        finite_float(rule.get("w_mbr"), 0.0) * finite_float(pair.get("mbr_z_delta"), 0.0)
        + finite_float(rule.get("w_a1"), 0.0) * finite_float(pair.get("A1_logprob_z_delta"), 0.0)
        + finite_float(rule.get("w_a2"), 0.0) * finite_float(pair.get("A2_logprob_z_delta"), 0.0)
        + finite_float(rule.get("w_a0"), 0.0) * finite_float(pair.get("A0_logprob_z_delta"), 0.0)
        + finite_float(rule.get("w_a5"), 0.0) * finite_float(pair.get("A5_logprob_z_delta"), 0.0)
        - finite_float(rule.get("w_len"), 0.0) * finite_float(pair.get("abs_length_delta_words"), 0.0)
        - finite_float(rule.get("w_edit"), 0.0) * finite_float(pair.get("edit_distance_to_current"), 0.0)
    )


def select_replacement(groups_rows: Sequence[Mapping[str, Any]], rule: Mapping[str, Any]) -> Tuple[Dict[str, Any] | None, float]:
    best_pair: Dict[str, Any] | None = None
    best_score = -math.inf
    for raw_pair in groups_rows:
        pair = dict(raw_pair)
        if not rule_passes(pair, rule):
            continue
        score = decision_score(pair, rule)
        if score > best_score:
            best_pair = pair
            best_score = score
    return best_pair, float(best_score)


def metrics_from_predictions(predictions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return prediction_metrics([dict(row) for row in predictions])


def current_prediction_from_pair(pair: Mapping[str, Any]) -> Dict[str, Any]:
    return pair_prediction_row(pair, use_candidate=False, selector="current_best")


def candidate_prediction_from_pair(pair: Mapping[str, Any], *, selector: str, score: float = 0.0) -> Dict[str, Any]:
    return pair_prediction_row(pair, use_candidate=True, selector=selector, score=score)


def prediction_map(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        key = prediction_sample_id(dict(row)) or sample_key(row, idx)
        out[key] = dict(row)
    return out


def row_word_edits(row: Mapping[str, Any]) -> int:
    if "word_edits" in row:
        return finite_int(row.get("word_edits"), 0)
    return int(word_edit_stats(prediction_to_reference(row), prediction_to_text(row))["edits"])


def row_char_edits(row: Mapping[str, Any]) -> int:
    if "char_edits" in row:
        return finite_int(row.get("char_edits"), 0)
    return int(char_edit_stats(prediction_to_reference(row), prediction_to_text(row))["edits"])


def reference_for_prediction(row: Mapping[str, Any], fallback: str = "") -> str:
    return prediction_to_reference(row) or fallback


def oracle_pair_for_group(group: Sequence[Mapping[str, Any]]) -> Dict[str, Any] | None:
    if not group:
        return None
    best: Dict[str, Any] | None = None
    best_key: Tuple[float, float, int] | None = None
    for idx, pair in enumerate(group):
        key = (
            finite_float(pair.get("candidate_wer"), 1e9),
            finite_float(pair.get("candidate_cer"), 1e9),
            idx,
        )
        if best_key is None or key < best_key:
            best = dict(pair)
            best_key = key
    return best


def aggregate_counts(values: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items(), key=lambda item: (-item[1], item[0])))
