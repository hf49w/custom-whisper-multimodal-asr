"""Add A12/A13 image-semantic rerank features to n-best candidates."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a9_candidate_utils import candidate_text, normalize_eval_text, read_jsonl, sample_id
from visspeech_custom_whisper_utils import resolve_cross_platform_path


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "with",
    "while",
}

VERBS = {
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

ADJECTIVES = {
    "black",
    "blue",
    "brown",
    "green",
    "large",
    "little",
    "old",
    "orange",
    "red",
    "small",
    "white",
    "yellow",
    "young",
}

SYNONYMS = {
    "bike": {"bicycle", "cycle"},
    "bicycle": {"bike", "cycle"},
    "boy": {"child", "kid"},
    "car": {"automobile", "vehicle"},
    "child": {"boy", "girl", "kid"},
    "dog": {"puppy", "canine"},
    "girl": {"child", "kid"},
    "man": {"person", "male"},
    "people": {"person", "persons"},
    "person": {"people", "man", "woman"},
    "puppy": {"dog"},
    "road": {"street"},
    "street": {"road", "sidewalk"},
    "woman": {"person", "female"},
}

SEMANTIC_FIELDS = [
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

SEMANTIC_LIST_FIELDS = [
    "changed_words",
    "deleted_top1_words",
    "supported_changed_words",
    "unsupported_changed_words",
    "deleted_supported_words",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--image-semantics-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--semantics-mode", choices=["true", "shuffle", "disable"], default="true")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        out: List[str] = []
        for item in value.values():
            out.extend(as_list(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(as_list(item))
        return out
    return [str(value)]


def words(text_or_values: Any) -> List[str]:
    values = as_list(text_or_values)
    tokens: List[str] = []
    for value in values:
        for token in normalize_eval_text(str(value)).split():
            if len(token) > 1 and token not in STOPWORDS:
                tokens.append(token)
    return tokens


def unique_words(text_or_values: Any) -> List[str]:
    seen = set()
    out: List[str] = []
    for token in words(text_or_values):
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def semantic_id(row: Dict[str, Any], index: int) -> str:
    return str(row.get("sample_id") or row.get("key") or row.get("utt_id") or row.get("id") or index)


def load_semantics(
    path: Path,
    *,
    mode: str,
    seed: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    rows = read_jsonl(path)
    packed_rows = [pack_semantics(row) if mode != "disable" else empty_semantics(row) for row in rows]
    if mode == "shuffle":
        random.Random(seed).shuffle(packed_rows)
    by_sample: Dict[str, Dict[str, Any]] = {}
    by_image: Dict[str, Dict[str, Any]] = {}
    if mode == "true":
        for index, row in enumerate(rows, start=1):
            packed = packed_rows[index - 1]
            sid = semantic_id(row, index)
            image_path = str(row.get("image_path") or "")
            by_sample[sid] = packed
            if image_path:
                by_image[image_path] = packed
    return by_sample, by_image, packed_rows


def empty_semantics(row: Dict[str, Any] | None = None) -> Dict[str, Any]:
    row = row or {}
    return {
        "sample_id": str(row.get("sample_id") or ""),
        "image_path": str(row.get("image_path") or ""),
        "captions": [],
        "caption_words": set(),
        "tags": set(),
        "nouns": set(),
        "verbs": set(),
        "adjectives": set(),
        "synonyms": set(),
        "all_visual_words": set(),
    }


def pack_semantics(row: Dict[str, Any]) -> Dict[str, Any]:
    captions = as_list(row.get("captions") or row.get("caption"))
    tag_payload = row.get("tags") or {}
    tag_words = set(unique_words(tag_payload))
    caption_words = set(unique_words(captions))
    objects = set(unique_words(tag_payload.get("objects") if isinstance(tag_payload, dict) else []))
    actions = set(unique_words(tag_payload.get("actions") if isinstance(tag_payload, dict) else []))
    attributes = set(unique_words(tag_payload.get("attributes") if isinstance(tag_payload, dict) else []))
    scenes = set(unique_words(tag_payload.get("scenes") if isinstance(tag_payload, dict) else []))
    all_visual = tag_words | caption_words
    verbs = actions | {word for word in all_visual if word in VERBS or word.endswith("ing")}
    adjectives = attributes | {word for word in all_visual if word in ADJECTIVES}
    nouns = (objects | scenes | all_visual) - verbs - adjectives
    synonyms: Set[str] = set()
    for word in all_visual:
        synonyms.update(SYNONYMS.get(word, set()))
    return {
        "sample_id": str(row.get("sample_id") or ""),
        "image_path": str(row.get("image_path") or ""),
        "captions": captions,
        "caption_words": caption_words,
        "tags": tag_words,
        "nouns": nouns,
        "verbs": verbs,
        "adjectives": adjectives,
        "synonyms": synonyms,
        "all_visual_words": all_visual | synonyms,
    }


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b) / len(a | b))


def overlap_fraction(candidate_words: Set[str], semantic_words: Set[str]) -> float:
    if not candidate_words:
        return 0.0
    return float(len(candidate_words & semantic_words) / len(candidate_words))


def edit_changed_words(base_words: Sequence[str], candidate_words: Sequence[str]) -> Dict[str, List[str]]:
    m, n = len(base_words), len(candidate_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    back = [[""] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
        back[i][0] = "del"
    for j in range(1, n + 1):
        dp[0][j] = j
        back[0][j] = "ins"
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if base_words[i - 1] == candidate_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "eq"
            else:
                choices = [
                    (dp[i - 1][j - 1] + 1, "sub"),
                    (dp[i][j - 1] + 1, "ins"),
                    (dp[i - 1][j] + 1, "del"),
                ]
                dp[i][j], back[i][j] = min(choices, key=lambda item: item[0])
    inserted_or_replaced: List[str] = []
    deleted: List[str] = []
    i, j = m, n
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "eq":
            i -= 1
            j -= 1
        elif op == "sub":
            inserted_or_replaced.append(candidate_words[j - 1])
            deleted.append(base_words[i - 1])
            i -= 1
            j -= 1
        elif op == "ins":
            inserted_or_replaced.append(candidate_words[j - 1])
            j -= 1
        else:
            deleted.append(base_words[i - 1])
            i -= 1
    return {
        "candidate_changed_words": list(reversed(inserted_or_replaced)),
        "deleted_top1_words": list(reversed(deleted)),
    }


def add_features(sample: Dict[str, Any], semantics: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(sample)
    candidates = [dict(candidate) for candidate in sample.get("candidates", [])]
    top1_words = words(candidate_text(candidates[0])) if candidates else []
    visual_words: Set[str] = set(semantics["all_visual_words"])
    nouns: Set[str] = set(semantics["nouns"])
    verbs: Set[str] = set(semantics["verbs"])
    adjectives: Set[str] = set(semantics["adjectives"])
    captions: Set[str] = set(semantics["caption_words"])
    tags: Set[str] = set(semantics["tags"])
    synonyms: Set[str] = set(semantics["synonyms"])

    for candidate in candidates:
        cand_words_list = words(candidate_text(candidate))
        cand_words = set(cand_words_list)
        changed = edit_changed_words(top1_words, cand_words_list)
        changed_words = [word for word in changed["candidate_changed_words"] if word not in STOPWORDS]
        deleted_words = [word for word in changed["deleted_top1_words"] if word not in STOPWORDS]
        supported = [word for word in changed_words if word in visual_words or word in synonyms]
        unsupported = [word for word in changed_words if word not in visual_words and word not in synonyms]
        deleted_supported = [word for word in deleted_words if word in visual_words or word in synonyms]
        denom = max(1, len(changed_words) + len(deleted_words))
        support_score = len(supported) / denom
        penalty_score = (len(unsupported) + len(deleted_supported)) / denom
        noun_verb_supported = [word for word in supported if word in nouns or word in verbs]

        caption_sim = jaccard(cand_words, captions)
        tag_overlap = overlap_fraction(cand_words, tags)
        candidate.update(
            {
                "caption_sim": caption_sim,
                "candidate_caption_similarity": caption_sim,
                "tag_overlap": tag_overlap,
                "candidate_tag_overlap": tag_overlap,
                "noun_overlap": overlap_fraction(cand_words, nouns),
                "verb_overlap": overlap_fraction(cand_words, verbs),
                "adjective_overlap": overlap_fraction(cand_words, adjectives),
                "synonym_overlap": overlap_fraction(cand_words, synonyms),
                "changed_words": changed_words,
                "deleted_top1_words": deleted_words,
                "supported_changed_words": supported,
                "unsupported_changed_words": unsupported,
                "deleted_supported_words": deleted_supported,
                "changed_word_support": float(support_score),
                "changed_word_penalty": float(penalty_score),
                "visual_gain": float(support_score - penalty_score),
                "supported_visual_change": int(bool(supported)),
                "supported_noun_verb_change": int(bool(noun_verb_supported)),
            }
        )
    out["candidates"] = candidates
    out["image_semantics_summary"] = {
        "captions": list(semantics["captions"]),
        "tag_count": len(tags),
        "noun_count": len(nouns),
        "verb_count": len(verbs),
        "adjective_count": len(adjectives),
    }
    return out


def zero_semantic_features(sample: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(sample)
    candidates = [dict(candidate) for candidate in sample.get("candidates", [])]
    for candidate in candidates:
        for field in SEMANTIC_FIELDS:
            candidate[field] = 0.0
        for field in SEMANTIC_LIST_FIELDS:
            candidate[field] = []
    out["candidates"] = candidates
    out["image_semantics_summary"] = {
        "captions": [],
        "tag_count": 0,
        "noun_count": 0,
        "verb_count": 0,
        "adjective_count": 0,
    }
    return out


def main() -> None:
    args = parse_args()
    samples = read_jsonl(resolve_cross_platform_path(args.input_jsonl))
    by_sample, by_image, semantic_sequence = load_semantics(
        resolve_cross_platform_path(args.image_semantics_jsonl),
        mode=args.semantics_mode,
        seed=args.shuffle_seed,
    )
    output_jsonl = resolve_cross_platform_path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    missing = 0
    for index, sample in enumerate(samples, start=1):
        sid = sample_id(sample)
        image_path = str(sample.get("image_path") or "")
        if args.semantics_mode == "shuffle" and semantic_sequence:
            semantics = semantic_sequence[(index - 1) % len(semantic_sequence)]
        else:
            semantics = by_sample.get(sid) or by_image.get(image_path)
        if semantics is None:
            semantics = empty_semantics({"sample_id": sid, "image_path": image_path})
            missing += 1
        enhanced = add_features(sample, semantics)
        if args.semantics_mode == "disable":
            enhanced = zero_semantic_features(enhanced)
        rows.append(enhanced)
        if index == 1 or index == len(samples) or index % max(1, args.log_every) == 0:
            print(f"[SEM-FEAT] row={index}/{len(samples)} missing_semantics={missing}")
    write_jsonl(output_jsonl, rows)
    print(json.dumps({"rows": len(rows), "missing_semantics": missing, "output_jsonl": str(output_jsonl)}, indent=2))


if __name__ == "__main__":
    main()
