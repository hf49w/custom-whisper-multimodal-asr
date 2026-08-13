"""Add normalized B2 reranking features to union n-best candidates."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a9_candidate_utils import candidate_text, mbr_scores, read_jsonl
from visspeech_custom_whisper_utils import normalize_eval_text, resolve_cross_platform_path


DEFAULT_TEACHER_FIELDS = ["A0_logprob", "A1_logprob", "A2_logprob", "A5_logprob"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--teacher-fields", default="A0_logprob,A1_logprob,A2_logprob,A5_logprob")
    parser.add_argument(
        "--zscore-fields",
        default="mbr_score,A0_logprob,A1_logprob,A2_logprob,A5_logprob,source_best_logprob,source_count",
    )
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_fields(text: str, default: Sequence[str]) -> List[str]:
    fields = [part.strip() for part in str(text or "").split(",") if part.strip()]
    return fields or list(default)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name.strip()).strip("_")


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def source_logprob(candidate: Dict[str, Any], model_name: str) -> float | None:
    source_scores = candidate.get("source_scores") or {}
    candidates = []
    for source, item in source_scores.items():
        source_text = str(source)
        if source_text == model_name or source_text.startswith(model_name):
            candidates.append(finite_float((item or {}).get("mean_logprob"), default=-math.inf))
    if candidates:
        return max(candidates)
    direct = candidate.get(f"source_{safe_name(model_name)}_logprob")
    if direct is not None:
        return finite_float(direct)
    return None


def feature_value(candidate: Dict[str, Any], field: str) -> float:
    if field in candidate and candidate[field] is not None:
        return finite_float(candidate[field])
    if field.endswith("_logprob"):
        model_name = field[: -len("_logprob")]
        value = source_logprob(candidate, model_name)
        if value is not None and math.isfinite(value):
            return float(value)
    return 0.0


def zscores(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    arr = np.asarray([float(value) for value in values], dtype=np.float32)
    std = float(arr.std())
    if std < 1e-8:
        return [0.0 for _ in values]
    mean = float(arr.mean())
    return [float((value - mean) / std) for value in arr]


def ensure_counts(candidate: Dict[str, Any]) -> None:
    normalized = str(candidate.get("normalized_text") or normalize_eval_text(candidate_text(candidate)))
    candidate["normalized_text"] = normalized
    words = normalized.split()
    candidate["word_count"] = int(candidate.get("word_count", len(words)) or len(words))
    candidate["char_count"] = int(candidate.get("char_count", len(normalized)) or len(normalized))


def add_features(sample: Dict[str, Any], teacher_fields: Sequence[str], zscore_fields: Sequence[str]) -> Dict[str, Any]:
    out = dict(sample)
    candidates = [dict(candidate) for candidate in sample.get("candidates", [])]
    if not candidates:
        out["candidates"] = []
        return out

    for candidate in candidates:
        ensure_counts(candidate)

    mbr = mbr_scores(candidates)
    for index, candidate in enumerate(candidates):
        candidate["mbr_score"] = float(candidate.get("mbr_score", mbr[index] if index < len(mbr) else 0.0))
        candidate["mbr"] = float(candidate["mbr_score"])

    top1 = candidates[0]
    top1_word_count = finite_float(top1.get("word_count"), default=0.0)
    top1_mbr = finite_float(top1.get("mbr_score"), default=0.0)
    mbr_z = zscores([feature_value(candidate, "mbr_score") for candidate in candidates])

    for index, candidate in enumerate(candidates):
        candidate["mbr_z"] = float(mbr_z[index])
        candidate["MBR_z"] = float(mbr_z[index])
        candidate["mbr_delta"] = float(feature_value(candidate, "mbr_score") - top1_mbr)
        candidate["MBR_delta"] = float(candidate["mbr_delta"])
        candidate["length_delta"] = float(finite_float(candidate.get("word_count"), default=0.0) - top1_word_count)
        candidate["length_abs_delta"] = abs(float(candidate["length_delta"]))

        sources = set(candidate.get("sources") or [])
        for source in sample.get("union_sources", []):
            candidate[f"source_{safe_name(str(source))}_present"] = 1 if source in sources else int(
                candidate.get(f"source_{safe_name(str(source))}_present", 0)
            )

    all_fields = list(dict.fromkeys(list(zscore_fields) + list(teacher_fields)))
    for field in all_fields:
        values = [feature_value(candidate, field) for candidate in candidates]
        z = zscores(values)
        top1_value = values[0] if values else 0.0
        for index, candidate in enumerate(candidates):
            candidate[field] = float(values[index])
            candidate[f"{field}_z"] = float(z[index])
            candidate[f"{field}_delta"] = float(values[index] - top1_value)

    out["candidates"] = candidates
    return out


def main() -> None:
    args = parse_args()
    input_jsonl = resolve_cross_platform_path(args.input_jsonl)
    output_jsonl = resolve_cross_platform_path(args.output_jsonl)
    samples = read_jsonl(input_jsonl)
    teacher_fields = parse_fields(args.teacher_fields, DEFAULT_TEACHER_FIELDS)
    zscore_fields = parse_fields(args.zscore_fields, [])
    rows: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        rows.append(add_features(sample, teacher_fields, zscore_fields))
        if args.log_every > 0 and (index == 1 or index == len(samples) or index % args.log_every == 0):
            print(f"[NORM-FEAT] row={index}/{len(samples)}")
    write_jsonl(output_jsonl, rows)
    print(
        json.dumps(
            {
                "input_jsonl": str(input_jsonl),
                "output_jsonl": str(output_jsonl),
                "rows": len(rows),
                "teacher_fields": teacher_fields,
                "zscore_fields": zscore_fields,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
