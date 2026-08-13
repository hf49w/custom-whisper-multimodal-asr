"""Build a source-union n-best list for B1/B2 reranking.

Each input is a dumped n-best JSONL from one decoding source, specified as
``SOURCE=path``. Candidates are deduplicated by normalized text per sample.
The output preserves source membership, original source ranks/logprobs, and
compact source indicator fields for downstream normalized feature building.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a9_candidate_utils import candidate_text, read_jsonl
from visspeech_custom_whisper_utils import normalize_eval_text, resolve_cross_platform_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Repeatable SOURCE=/path/to/nbest.jsonl input. First input is the default primary source.",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--primary-source", default="")
    parser.add_argument(
        "--include-all-samples",
        action="store_true",
        help="Include sample IDs that appear only in non-primary sources.",
    )
    return parser.parse_args()


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name.strip()).strip("_")


def parse_source_spec(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"--input must be SOURCE=PATH, got: {raw}")
    source, path = raw.split("=", 1)
    source = source.strip()
    if not source:
        raise ValueError(f"Empty source name in --input {raw!r}")
    return source, resolve_cross_platform_path(path.strip())


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_key(sample: Dict[str, Any], index: int) -> str:
    return str(sample.get("sample_id") or sample.get("key") or sample.get("utt_id") or index)


def candidate_norm(candidate: Dict[str, Any]) -> str:
    norm = str(candidate.get("normalized_text") or "").strip()
    if norm:
        return norm
    return normalize_eval_text(candidate_text(candidate))


def numeric(candidate: Dict[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in candidate and candidate[name] is not None:
            try:
                value = float(candidate[name])
                if math.isfinite(value):
                    return value
            except Exception:
                pass
    return float(default)


def source_record(candidate: Dict[str, Any], source: str, source_index: int) -> Dict[str, Any]:
    rank = int(candidate.get("beam_rank", source_index))
    logprob = numeric(candidate, "asr_mean_logprob", "mean_logprob", "avg_token_logprob", default=0.0)
    return {
        "source": source,
        "source_index": int(source_index),
        "beam_rank": rank,
        "mean_logprob": logprob,
        "asr_mean_logprob": numeric(candidate, "asr_mean_logprob", "mean_logprob", default=logprob),
        "sum_logprob": numeric(candidate, "sum_logprob", default=0.0),
        "avg_token_logprob": numeric(candidate, "avg_token_logprob", "mean_logprob", default=logprob),
        "min_token_logprob": numeric(candidate, "min_token_logprob", default=0.0),
        "token_count": int(float(candidate.get("token_count", 0) or 0)),
        "word_count": int(float(candidate.get("word_count", 0) or 0)),
        "char_count": int(float(candidate.get("char_count", 0) or 0)),
        "compression_ratio": numeric(candidate, "compression_ratio", default=0.0),
    }


def best_source_item(candidate: Dict[str, Any], priority: Dict[str, int]) -> Dict[str, Any]:
    items = list((candidate.get("source_scores") or {}).values())
    if not items:
        return {}
    return min(
        items,
        key=lambda item: (
            priority.get(str(item.get("source", "")), 10_000),
            int(item.get("beam_rank", 10_000)),
            -float(item.get("mean_logprob", -1e9)),
        ),
    )


def finalize_candidate(candidate: Dict[str, Any], source_names: Sequence[str], priority: Dict[str, int]) -> Dict[str, Any]:
    source_scores = candidate.get("source_scores") or {}
    source_items = list(source_scores.values())
    best = best_source_item(candidate, priority)
    sources = sorted(source_scores.keys(), key=lambda name: priority.get(name, 10_000))
    min_rank = min((int(item.get("beam_rank", 10_000)) for item in source_items), default=10_000)
    best_logprob = max((float(item.get("mean_logprob", -1e9)) for item in source_items), default=0.0)
    out = {
        key: value
        for key, value in candidate.items()
        if key not in {"source_scores"}
    }
    out.update(
        {
            "sources": sources,
            "source_scores": source_scores,
            "source_count": len(sources),
            "source_min_rank": int(min_rank),
            "source_best_logprob": float(best_logprob),
            "best_source": str(best.get("source", sources[0] if sources else "")),
            "mean_logprob": float(best.get("mean_logprob", best_logprob)),
            "asr_mean_logprob": float(best.get("asr_mean_logprob", best.get("mean_logprob", best_logprob))),
            "sum_logprob": float(best.get("sum_logprob", 0.0)),
            "avg_token_logprob": float(best.get("avg_token_logprob", best.get("mean_logprob", best_logprob))),
            "min_token_logprob": float(best.get("min_token_logprob", 0.0)),
            "token_count": int(best.get("token_count", out.get("token_count", 0)) or 0),
            "word_count": int(best.get("word_count", len(str(out.get("normalized_text", "")).split())) or 0),
            "char_count": int(best.get("char_count", len(str(out.get("normalized_text", "")))) or 0),
            "compression_ratio": float(best.get("compression_ratio", out.get("compression_ratio", 0.0)) or 0.0),
        }
    )
    for source in source_names:
        safe = safe_name(source)
        item = source_scores.get(source) or {}
        out[f"source_{safe}_present"] = 1 if item else 0
        out[f"source_{safe}_rank"] = int(item.get("beam_rank", 10_000)) if item else 10_000
        out[f"source_{safe}_logprob"] = float(item.get("mean_logprob", 0.0)) if item else 0.0
    return out


def candidate_sort_key(candidate: Dict[str, Any], primary_source: str, priority: Dict[str, int]) -> tuple:
    scores = candidate.get("source_scores") or {}
    primary = scores.get(primary_source) or {}
    primary_rank = int(primary.get("beam_rank", 10_000))
    best = best_source_item(candidate, priority)
    min_source_priority = min((priority.get(name, 10_000) for name in scores), default=10_000)
    return (
        primary_rank,
        int(best.get("beam_rank", 10_000)),
        min_source_priority,
        -len(scores),
        -float(candidate.get("source_best_logprob", best.get("mean_logprob", -1e9))),
        str(candidate.get("normalized_text", "")),
    )


def main() -> None:
    args = parse_args()
    specs = [parse_source_spec(raw) for raw in args.input]
    source_names = [source for source, _path in specs]
    priority = {source: index for index, source in enumerate(source_names)}
    primary_source = args.primary_source or source_names[0]
    if primary_source not in priority:
        raise ValueError(f"--primary-source {primary_source!r} is not one of {source_names}")

    grouped: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for source, path in specs:
        rows = read_jsonl(path)
        for row_index, sample in enumerate(rows, start=1):
            key = sample_key(sample, row_index)
            if key not in grouped:
                grouped[key] = {
                    "sample": sample,
                    "source_samples": {},
                    "candidates": {},
                }
                order.append(key)
            grouped[key]["source_samples"][source] = sample
            if source == primary_source:
                grouped[key]["sample"] = sample
            for cand_index, candidate in enumerate(sample.get("candidates", [])):
                norm = candidate_norm(candidate)
                if not norm:
                    continue
                union_candidate = grouped[key]["candidates"].get(norm)
                if union_candidate is None:
                    union_candidate = {
                        "text": str(candidate.get("text") or candidate.get("normalized_text") or ""),
                        "normalized_text": norm,
                        "source_scores": {},
                    }
                    grouped[key]["candidates"][norm] = union_candidate
                item = source_record(candidate, source, cand_index)
                old = union_candidate["source_scores"].get(source)
                if old is None or (
                    int(item["beam_rank"]),
                    -float(item["mean_logprob"]),
                ) < (
                    int(old.get("beam_rank", 10_000)),
                    -float(old.get("mean_logprob", -1e9)),
                ):
                    union_candidate["source_scores"][source] = item
                    if priority[source] <= priority.get(str(union_candidate.get("best_source", source)), 10_000):
                        union_candidate["text"] = str(candidate.get("text") or union_candidate["text"])

    output_rows: List[Dict[str, Any]] = []
    for key in order:
        record = grouped[key]
        if not args.include_all_samples and primary_source not in record["source_samples"]:
            continue
        sample = record["sample"]
        candidates = [
            finalize_candidate(candidate, source_names, priority)
            for candidate in record["candidates"].values()
        ]
        candidates.sort(key=lambda candidate: candidate_sort_key(candidate, primary_source, priority))
        candidates = candidates[: max(1, int(args.top_k))]
        for rank, candidate in enumerate(candidates):
            candidate["beam_rank"] = rank
            candidate["union_rank"] = rank
        output_rows.append(
            {
                "sample_id": key,
                "key": str(sample.get("key") or key),
                "audio_path": str(sample.get("audio_path") or sample.get("wav_path") or ""),
                "wav_path": str(sample.get("wav_path") or sample.get("audio_path") or ""),
                "image_path": str(sample.get("image_path") or ""),
                "reference": str(sample.get("reference") or sample.get("ref_text") or sample.get("annotation") or sample.get("text") or ""),
                "normalized_reference": str(sample.get("normalized_reference") or ""),
                "union_sources": source_names,
                "primary_source": primary_source,
                "max_candidates": int(args.top_k),
                "candidates": candidates,
            }
        )

    output_jsonl = resolve_cross_platform_path(args.output_jsonl)
    write_jsonl(output_jsonl, output_rows)
    print(
        json.dumps(
            {
                "output_jsonl": str(output_jsonl),
                "rows": len(output_rows),
                "sources": source_names,
                "primary_source": primary_source,
                "top_k": int(args.top_k),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
