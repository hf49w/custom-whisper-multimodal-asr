"""Shared utilities for A9 n-best candidate reranking experiments."""

from __future__ import annotations

import csv
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom_whisper.utils import compression_ratio
from visspeech_custom_whisper_utils import (
    compute_cer,
    compute_wer,
    normalize_eval_text,
    resolve_cross_platform_path,
    write_jsonl,
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def save_pickle(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dict payload in {path}")
    return payload


def edit_distance(seq_a: Sequence[Any], seq_b: Sequence[Any]) -> int:
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)
    prev = list(range(len(seq_b) + 1))
    for i, token_a in enumerate(seq_a, start=1):
        current = [i]
        for j, token_b in enumerate(seq_b, start=1):
            cost = 0 if token_a == token_b else 1
            current.append(min(prev[j] + 1, current[j - 1] + 1, prev[j - 1] + cost))
        prev = current
    return prev[-1]


def word_edit_stats(ref_text: str, hyp_text: str) -> Dict[str, Any]:
    ref_words = normalize_eval_text(ref_text).split()
    hyp_words = normalize_eval_text(hyp_text).split()
    edits = edit_distance(ref_words, hyp_words)
    denom = max(1, len(ref_words))
    return {"edits": edits, "denom": denom, "wer": edits / denom}


def char_edit_stats(ref_text: str, hyp_text: str) -> Dict[str, Any]:
    ref_chars = list(normalize_eval_text(ref_text))
    hyp_chars = list(normalize_eval_text(hyp_text))
    edits = edit_distance(ref_chars, hyp_chars)
    denom = max(1, len(ref_chars))
    return {"edits": edits, "denom": denom, "cer": edits / denom}


def prediction_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    refs = [str(row["reference"]) for row in rows]
    hyps = [str(row["prediction"]) for row in rows]
    return {"count": len(rows), "wer": compute_wer(refs, hyps), "cer": compute_cer(refs, hyps)}


def make_prediction_row(
    sample: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    selected_index: int,
    selector: str,
) -> Dict[str, Any]:
    ref = str(sample.get("reference", sample.get("ref_text", "")))
    hyp = str(candidate.get("text", ""))
    word = word_edit_stats(ref, hyp)
    char = char_edit_stats(ref, hyp)
    return {
        "sample_id": str(sample.get("sample_id", sample.get("key", ""))),
        "key": str(sample.get("key", sample.get("sample_id", ""))),
        "audio_path": str(sample.get("audio_path", sample.get("wav_path", ""))),
        "wav_path": str(sample.get("wav_path", sample.get("audio_path", ""))),
        "image_path": str(sample.get("image_path", "")),
        "reference": ref,
        "normalized_reference": normalize_eval_text(ref),
        "prediction": hyp,
        "normalized_prediction": normalize_eval_text(hyp),
        "selected_index": int(selected_index),
        "beam_rank": int(candidate.get("beam_rank", selected_index)),
        "selector": selector,
        "word_edits": word["edits"],
        "word_denom": word["denom"],
        "sample_wer": word["wer"],
        "char_edits": char["edits"],
        "char_denom": char["denom"],
        "sample_cer": char["cer"],
    }


def parse_float_grid(text: str, default: Sequence[float]) -> List[float]:
    if not text:
        return [float(value) for value in default]
    values = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise ValueError("Grid must contain at least one value")
    return values


def candidate_text(candidate: Dict[str, Any]) -> str:
    return str(candidate.get("text") or candidate.get("normalized_text") or "").strip()


def sample_id(sample: Dict[str, Any]) -> str:
    return str(sample.get("sample_id") or sample.get("key") or sample.get("utt_id") or "")


def prediction_sample_id(prediction: Dict[str, Any]) -> str:
    return str(
        prediction.get("sample_id")
        or prediction.get("key")
        or prediction.get("utt_id")
        or prediction.get("id")
        or ""
    )


def prediction_text(prediction: Dict[str, Any]) -> str:
    return str(
        prediction.get("prediction")
        or prediction.get("pred_text")
        or prediction.get("hyp")
        or prediction.get("text")
        or ""
    )


def prediction_reference(prediction: Dict[str, Any]) -> str:
    return str(
        prediction.get("reference")
        or prediction.get("ref_text")
        or prediction.get("ref")
        or ""
    )


def map_predictions(predictions: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mapped: Dict[str, Dict[str, Any]] = {}
    for index, prediction in enumerate(predictions):
        key = prediction_sample_id(prediction) or str(index)
        mapped[key] = dict(prediction)
    return mapped


def candidate_by_prediction(sample: Dict[str, Any], prediction: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    candidates = sample.get("candidates", [])
    if not candidates:
        return 0, {}
    if "selected_index" in prediction:
        index = max(0, min(int(prediction["selected_index"]), len(candidates) - 1))
        return index, candidates[index]
    if "beam_rank" in prediction:
        beam_rank = int(prediction["beam_rank"])
        for index, candidate in enumerate(candidates):
            if int(candidate.get("beam_rank", index)) == beam_rank:
                return index, candidate
    normalized = normalize_eval_text(prediction_text(prediction))
    for index, candidate in enumerate(candidates):
        if normalize_eval_text(candidate_text(candidate)) == normalized:
            return index, candidate
    return 0, candidates[0]


def normalized_word_distance(text_a: str, text_b: str) -> float:
    words_a = normalize_eval_text(text_a).split()
    words_b = normalize_eval_text(text_b).split()
    denom = max(1, len(words_a), len(words_b))
    return edit_distance(words_a, words_b) / denom


def mbr_scores(candidates: Sequence[Dict[str, Any]]) -> List[float]:
    tokenized = [
        normalize_eval_text(candidate_text(candidate)).split()
        for candidate in candidates
    ]
    scores: List[float] = []
    for i, words_i in enumerate(tokenized):
        distances = []
        for j, words_j in enumerate(tokenized):
            if i == j:
                continue
            denom = max(1, len(words_i), len(words_j))
            distances.append(edit_distance(words_i, words_j) / denom)
        avg_distance = float(np.mean(distances)) if distances else 0.0
        scores.append(-avg_distance)
    return scores


def length_scores(candidates: Sequence[Dict[str, Any]]) -> List[float]:
    lengths = np.asarray(
        [float(candidate.get("word_count", 0.0)) for candidate in candidates],
        dtype=np.float32,
    )
    if lengths.size == 0:
        return []
    median = float(np.median(lengths))
    denom = max(1.0, median)
    return [float(-abs(length - median) / denom) for length in lengths]


class ClipCandidateScorer:
    def __init__(self, model_name: str, device: torch.device, no_download: bool = True):
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError("CLIP scoring requires transformers with CLIPModel support") from exc

        self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=no_download)
        self.model = CLIPModel.from_pretrained(model_name, local_files_only=no_download).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def score_sample(self, image_path: str, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=list(texts),
            images=[image] * len(texts),
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        image_features = F.normalize(outputs.image_embeds, dim=-1)
        text_features = F.normalize(outputs.text_embeds, dim=-1)
        return (image_features * text_features).sum(dim=-1).float().cpu().tolist()


def zscores(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    std = float(arr.std())
    if std < 1e-8:
        return [0.0 for _ in values]
    mean = float(arr.mean())
    return [float((value - mean) / std) for value in arr]


def ensure_candidate_scores(
    samples: Sequence[Dict[str, Any]],
    *,
    clip_scorer: Optional[ClipCandidateScorer] = None,
    log_every: int = 100,
) -> List[Dict[str, Any]]:
    """Return a deep-ish scored copy with MBR/length/CLIP fields populated."""

    scored_samples: List[Dict[str, Any]] = []
    for sample_index, sample in enumerate(samples, start=1):
        out_sample = dict(sample)
        candidates = [dict(candidate) for candidate in sample.get("candidates", [])]
        mbr = mbr_scores(candidates)
        length = length_scores(candidates)
        for idx, candidate in enumerate(candidates):
            candidate["mbr_score"] = float(mbr[idx]) if idx < len(mbr) else 0.0
            candidate["length_score"] = float(length[idx]) if idx < len(length) else 0.0
            if "asr_mean_logprob" not in candidate:
                candidate["asr_mean_logprob"] = float(candidate.get("mean_logprob", 0.0))

        if clip_scorer is not None:
            texts = [candidate_text(candidate) for candidate in candidates]
            clip = clip_scorer.score_sample(str(sample.get("image_path", "")), texts)
            clip_z = zscores(clip)
            for idx, candidate in enumerate(candidates):
                candidate["clip_score"] = float(clip[idx]) if idx < len(clip) else 0.0
                candidate["clip_zscore"] = float(clip_z[idx]) if idx < len(clip_z) else 0.0
        else:
            clip_values = [float(candidate.get("clip_score", 0.0)) for candidate in candidates]
            clip_z = zscores(clip_values)
            for idx, candidate in enumerate(candidates):
                candidate["clip_score"] = clip_values[idx] if idx < len(clip_values) else 0.0
                candidate["clip_zscore"] = float(candidate.get("clip_zscore", clip_z[idx] if idx < len(clip_z) else 0.0))

        out_sample["candidates"] = candidates
        scored_samples.append(out_sample)
        if log_every > 0 and (sample_index == 1 or sample_index % log_every == 0 or sample_index == len(samples)):
            print(f"[SCORE] sample={sample_index}/{len(samples)} candidates={len(candidates)}")
    return scored_samples


def select_by_score(
    sample: Dict[str, Any],
    *,
    a: float,
    b: float,
    c: float,
    d: float,
    extra_weights: Optional[Dict[str, float]] = None,
) -> Tuple[int, Dict[str, Any], float]:
    candidates = sample.get("candidates", [])
    if not candidates:
        raise ValueError(f"Sample has no candidates: {sample.get('sample_id')}")
    best_index = 0
    best_score = -math.inf
    for idx, candidate in enumerate(candidates):
        score = (
            float(a) * float(candidate.get("asr_mean_logprob", candidate.get("mean_logprob", 0.0)))
            + float(b) * float(candidate.get("clip_zscore", 0.0))
            + float(c) * float(candidate.get("mbr_score", 0.0))
            + float(d) * float(candidate.get("length_score", 0.0))
        )
        for field_name, weight in (extra_weights or {}).items():
            score += float(weight) * float(candidate.get(field_name, 0.0))
        if score > best_score:
            best_score = score
            best_index = idx
    return best_index, candidates[best_index], best_score


def oracle_index(sample: Dict[str, Any], limit: Optional[int] = None) -> int:
    candidates = list(sample.get("candidates", []))
    if limit is not None:
        candidates = candidates[: max(1, int(limit))]
    if not candidates:
        return 0
    ref = str(sample.get("reference", ""))
    best_index = 0
    best_key: Optional[Tuple[int, int, int]] = None
    for idx, candidate in enumerate(candidates):
        word = word_edit_stats(ref, candidate_text(candidate))
        char = char_edit_stats(ref, candidate_text(candidate))
        key = (int(word["edits"]), int(char["edits"]), idx)
        if best_key is None or key < best_key:
            best_key = key
            best_index = idx
    return best_index


def predictions_for_indices(
    samples: Sequence[Dict[str, Any]],
    indices: Sequence[int],
    *,
    selector: str,
) -> List[Dict[str, Any]]:
    predictions = []
    for sample, index in zip(samples, indices):
        candidates = sample.get("candidates", [])
        if not candidates:
            continue
        safe_index = max(0, min(int(index), len(candidates) - 1))
        predictions.append(
            make_prediction_row(
                sample,
                candidates[safe_index],
                selected_index=safe_index,
                selector=selector,
            )
        )
    return predictions


def oracle_curve(samples: Sequence[Dict[str, Any]], ks: Sequence[int]) -> Dict[str, Dict[str, Any]]:
    curve: Dict[str, Dict[str, Any]] = {}
    for k in ks:
        indices = [oracle_index(sample, limit=k) for sample in samples]
        predictions = predictions_for_indices(samples, indices, selector=f"oracle_top{k}")
        curve[str(k)] = prediction_metrics(predictions)
    return curve


def candidate_features(candidate: Dict[str, Any]) -> Dict[str, float]:
    return {
        "beam_rank": float(candidate.get("beam_rank", 0.0)),
        "sum_logprob": float(candidate.get("sum_logprob", 0.0)),
        "mean_logprob": float(candidate.get("mean_logprob", candidate.get("asr_mean_logprob", 0.0))),
        "min_token_logprob": float(candidate.get("min_token_logprob", 0.0)),
        "token_count": float(candidate.get("token_count", 0.0)),
        "word_count": float(candidate.get("word_count", 0.0)),
        "char_count": float(candidate.get("char_count", 0.0)),
        "compression_ratio": float(candidate.get("compression_ratio", 0.0)),
        "mbr_score": float(candidate.get("mbr_score", 0.0)),
        "clip_score": float(candidate.get("clip_score", 0.0)),
        "clip_zscore": float(candidate.get("clip_zscore", 0.0)),
        "caption_candidate_clip_score": float(candidate.get("caption_candidate_clip_score", 0.0)),
        "tag_overlap": float(candidate.get("tag_overlap", 0.0)),
        "length_score": float(candidate.get("length_score", 0.0)),
    }


DEFAULT_FEATURE_NAMES = [
    "beam_rank",
    "sum_logprob",
    "mean_logprob",
    "min_token_logprob",
    "token_count",
    "word_count",
    "char_count",
    "compression_ratio",
    "mbr_score",
    "clip_score",
    "clip_zscore",
    "caption_candidate_clip_score",
    "tag_overlap",
    "length_score",
]


def flatten_candidate_table(
    samples: Sequence[Dict[str, Any]],
    *,
    feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES,
    with_labels: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]], List[str]]:
    rows: List[List[float]] = []
    labels: List[float] = []
    groups: List[int] = []
    index_pairs: List[Tuple[int, int]] = []
    sample_ids: List[str] = []
    for sample_idx, sample in enumerate(samples):
        sample_id = str(sample.get("sample_id", sample_idx))
        ref = str(sample.get("reference", ""))
        for cand_idx, candidate in enumerate(sample.get("candidates", [])):
            features = candidate_features(candidate)
            rows.append([float(features.get(name, 0.0)) for name in feature_names])
            if with_labels:
                labels.append(-float(word_edit_stats(ref, candidate_text(candidate))["wer"]))
            else:
                labels.append(0.0)
            groups.append(sample_idx)
            index_pairs.append((sample_idx, cand_idx))
            sample_ids.append(sample_id)
    if not rows:
        raise ValueError("No candidates found")
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(groups, dtype=np.int64),
        index_pairs,
        sample_ids,
    )


def select_model_predictions(
    samples: Sequence[Dict[str, Any]],
    scores: np.ndarray,
    index_pairs: Sequence[Tuple[int, int]],
    *,
    selector: str,
) -> List[Dict[str, Any]]:
    best_by_sample: Dict[int, Tuple[int, float]] = {}
    for score, (sample_idx, cand_idx) in zip(scores, index_pairs):
        score_value = float(score)
        old = best_by_sample.get(sample_idx)
        if old is None or score_value > old[1]:
            best_by_sample[sample_idx] = (cand_idx, score_value)
    indices = [best_by_sample.get(i, (0, 0.0))[0] for i in range(len(samples))]
    predictions = predictions_for_indices(samples, indices, selector=selector)
    for prediction, sample_idx in zip(predictions, range(len(predictions))):
        prediction["reranker_score"] = float(best_by_sample.get(sample_idx, (0, 0.0))[1])
    return predictions


def write_predictions(path: Path, predictions: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, predictions)


def score_candidate(
    candidate: Dict[str, Any],
    *,
    a: float,
    b: float,
    c: float,
    d: float,
    extra_weights: Optional[Dict[str, float]] = None,
) -> float:
    score = (
        float(a) * float(candidate.get("asr_mean_logprob", candidate.get("mean_logprob", 0.0)))
        + float(b) * float(candidate.get("clip_zscore", 0.0))
        + float(c) * float(candidate.get("mbr_score", 0.0))
        + float(d) * float(candidate.get("length_score", 0.0))
    )
    for field_name, weight in (extra_weights or {}).items():
        score += float(weight) * float(candidate.get(field_name, 0.0))
    return float(score)


def parse_extra_weight_specs(fields_text: str, specs_text: str) -> List[Dict[str, float]]:
    fields = [part.strip() for part in fields_text.split(",") if part.strip()]
    if not fields:
        return [{}]
    if not specs_text:
        return [{field: 0.0 for field in fields}]
    specs: List[Dict[str, float]] = []
    for raw_spec in specs_text.split(";"):
        raw_spec = raw_spec.strip()
        if not raw_spec:
            continue
        if "=" in raw_spec:
            weights = {field: 0.0 for field in fields}
            for item in raw_spec.split(","):
                if not item.strip():
                    continue
                key, value = item.split("=", 1)
                key = key.strip()
                if key not in weights:
                    raise ValueError(f"Unknown extra score field in spec: {key}")
                weights[key] = float(value)
            specs.append(weights)
        else:
            values = [float(part.strip()) for part in raw_spec.split(",") if part.strip()]
            if len(values) != len(fields):
                raise ValueError(
                    f"Extra weight spec {raw_spec!r} has {len(values)} values, expected {len(fields)}"
                )
            specs.append(dict(zip(fields, values)))
    if not specs:
        raise ValueError("--extra-score-weight-specs did not produce any specs")
    return specs


def pruned_candidate_indices(
    sample: Dict[str, Any],
    *,
    asr_top_n: int,
    mbr_top_m: int,
    clip_top_c: int,
    edit_to_top1_max: float,
) -> List[int]:
    candidates = list(sample.get("candidates", []))
    if not candidates:
        return []
    keep = set(range(min(max(0, asr_top_n), len(candidates))))
    if mbr_top_m > 0:
        keep.update(
            sorted(
                range(len(candidates)),
                key=lambda idx: float(candidates[idx].get("mbr_score", 0.0)),
                reverse=True,
            )[:mbr_top_m]
        )
    if clip_top_c > 0:
        keep.update(
            sorted(
                range(len(candidates)),
                key=lambda idx: float(candidates[idx].get("clip_zscore", candidates[idx].get("clip_score", 0.0))),
                reverse=True,
            )[:clip_top_c]
        )
    top1_text = candidate_text(candidates[0])
    for idx, candidate in enumerate(candidates):
        if normalized_word_distance(top1_text, candidate_text(candidate)) <= float(edit_to_top1_max):
            keep.add(idx)
    return sorted(keep)


def select_by_score_from_indices(
    sample: Dict[str, Any],
    candidate_indices: Sequence[int],
    *,
    a: float,
    b: float,
    c: float,
    d: float,
    extra_weights: Optional[Dict[str, float]] = None,
) -> Tuple[int, Dict[str, Any], float]:
    candidates = sample.get("candidates", [])
    if not candidates:
        raise ValueError(f"Sample has no candidates: {sample.get('sample_id')}")
    best_index = 0
    best_score = -math.inf
    for idx in candidate_indices or [0]:
        if idx < 0 or idx >= len(candidates):
            continue
        score = score_candidate(
            candidates[idx],
            a=a,
            b=b,
            c=c,
            d=d,
            extra_weights=extra_weights,
        )
        if score > best_score:
            best_index = idx
            best_score = score
    return best_index, candidates[best_index], best_score


def numeric_summary(values: Sequence[float]) -> Dict[str, Any]:
    clean = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))])
    if clean.size == 0:
        return {"count": 0}
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "median": float(np.median(clean)),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "p90": float(np.percentile(clean, 90)),
    }


def categorical_counts(values: Sequence[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))
