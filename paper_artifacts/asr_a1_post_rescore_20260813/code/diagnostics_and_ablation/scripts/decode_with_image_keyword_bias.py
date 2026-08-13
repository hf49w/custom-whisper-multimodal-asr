"""A14 decode with image-keyword token logits bias.

This script does not train ASR models and does not put image captions/tags into
the Whisper prompt. It extracts keywords from an image_semantics JSONL file and
adds a small bias to matching keyword token IDs during decoding.

Run true/shuffle/disable keyword modes on validation to select beta, then run
the selected beta once on test.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import custom_whisper
from a9_candidate_utils import normalize_eval_text, prediction_metrics, read_jsonl, word_edit_stats, char_edit_stats
from custom_whisper.decoding import DecodingTask, LogitFilter
from eval_decode_only_oracle_rerank import load_eval_model, model_needs_images
from visspeech_custom_whisper_utils import build_tokenizer_and_prefix, load_manifest, resolve_cross_platform_path


STOPWORDS = {"a", "an", "and", "are", "at", "in", "is", "of", "on", "the", "to", "with"}


class KeywordBiasFilter(LogitFilter):
    def __init__(self, token_ids: Sequence[int], beta: float):
        self.token_ids = sorted(set(int(token_id) for token_id in token_ids))
        self.beta = float(beta)

    def apply(self, logits: torch.Tensor, tokens: torch.Tensor) -> None:
        if self.token_ids and self.beta != 0.0:
            logits[:, self.token_ids] += self.beta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint-path", default="")
    source.add_argument("--whisper-model", default="")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--image-semantics-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keyword-mode", choices=["true", "shuffle", "disable"], default="true")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--beta-values", default="0.02,0.05,0.1")
    parser.add_argument("--max-keywords", type=int, default=12)
    parser.add_argument("--device", default="")
    parser.add_argument("--beam-size", type=int, default=20, choices=[10, 20, 30, 50])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--disable-images", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.set_defaults(no_download=True)
    parser.add_argument("--no-download", dest="no_download", action="store_true")
    parser.add_argument("--allow-download", dest="no_download", action="store_false")
    parser.add_argument("--visual-model-name", default="")
    return parser.parse_args()


def parse_float_values(text: str) -> List[float]:
    return [float(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def keywords_from_semantics(row: Dict[str, Any], max_keywords: int) -> List[str]:
    values = as_list(row.get("tags")) + as_list(row.get("captions"))
    seen: Set[str] = set()
    out: List[str] = []
    for value in values:
        for token in normalize_eval_text(str(value)).split():
            if len(token) <= 2 or token in STOPWORDS or token in seen:
                continue
            seen.add(token)
            out.append(token)
            if len(out) >= max_keywords:
                return out
    return out


def load_keyword_rows(path: Path, *, mode: str, seed: int, max_keywords: int) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    packed = [
        {
            "sample_id": str(row.get("sample_id") or row.get("key") or row.get("utt_id") or index),
            "image_path": str(row.get("image_path") or ""),
            "keywords": keywords_from_semantics(row, max_keywords),
        }
        for index, row in enumerate(rows, start=1)
    ]
    if mode == "shuffle":
        random.Random(seed).shuffle(packed)
    elif mode == "disable":
        for row in packed:
            row["keywords"] = []
    return packed


def sample_id_for_row(row: Dict[str, Any], index: int) -> str:
    wav_path = str(row.get("wav_path", ""))
    return str(row.get("key") or row.get("utt_id") or Path(wav_path).stem or index)


def keyword_token_ids(keywords: Sequence[str], tokenizer: Any) -> List[int]:
    token_ids: List[int] = []
    for keyword in keywords:
        for token in tokenizer.encode(" " + keyword):
            token_ids.append(int(token))
    return sorted(set(token_ids))


@torch.no_grad()
def decode_one(
    model: Any,
    mel: torch.Tensor,
    options: Any,
    *,
    keyword_tokens: Sequence[int],
    beta: float,
    image_path: str,
    use_images: bool,
) -> Any:
    task = DecodingTask(model, options)
    task.logit_filters.append(KeywordBiasFilter(keyword_tokens, beta))
    if use_images and hasattr(model, "use_visual_context"):
        with model.use_visual_context(image=image_path):
            return task.run(mel.unsqueeze(0))[0]
    return task.run(mel.unsqueeze(0))[0]


def prediction_row(row: Dict[str, Any], index: int, text: str, *, beta: float, keywords: Sequence[str], mode: str) -> Dict[str, Any]:
    reference = str(row.get("annotation", row.get("text", "")))
    word = word_edit_stats(reference, text)
    char = char_edit_stats(reference, text)
    return {
        "sample_id": sample_id_for_row(row, index),
        "key": str(row.get("key") or sample_id_for_row(row, index)),
        "audio_path": str(row.get("wav_path", row.get("audio_path", ""))),
        "image_path": str(row.get("image_path", "")),
        "reference": reference,
        "prediction": str(text),
        "normalized_reference": normalize_eval_text(reference),
        "normalized_prediction": normalize_eval_text(text),
        "beta": float(beta),
        "keyword_mode": mode,
        "keywords": list(keywords),
        "word_edits": word["edits"],
        "word_denom": word["denom"],
        "sample_wer": word["wer"],
        "char_edits": char["edits"],
        "char_denom": char["denom"],
        "sample_cer": char["cer"],
    }


def main() -> None:
    args = parse_args()
    output_dir = resolve_cross_platform_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_cross_platform_path(args.manifest_path)
    rows = load_manifest(manifest_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError(f"No rows loaded from {manifest_path}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, model_source, is_checkpoint = load_eval_model(args, device)
    tokenizer, _initial_tokens = build_tokenizer_and_prefix(model)
    use_images = model_needs_images(model, is_checkpoint=is_checkpoint, disable_images=args.disable_images)
    options = custom_whisper.DecodingOptions(
        language="en",
        task="transcribe",
        beam_size=int(args.beam_size),
        fp16=device.type == "cuda",
        without_timestamps=True,
    )

    semantic_rows = load_keyword_rows(
        resolve_cross_platform_path(args.image_semantics_jsonl),
        mode=args.keyword_mode,
        seed=args.shuffle_seed,
        max_keywords=args.max_keywords,
    )
    betas = parse_float_values(args.beta_values)
    all_metrics: Dict[str, Any] = {}
    started = time.time()
    for beta in betas:
        predictions: List[Dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            image_path = str(row.get("image_path", ""))
            keywords = semantic_rows[(index - 1) % len(semantic_rows)]["keywords"] if semantic_rows else []
            token_ids = keyword_token_ids(keywords, tokenizer)
            audio = custom_whisper.pad_or_trim(custom_whisper.load_audio(str(row["wav_path"])))
            mel = custom_whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(device)
            result = decode_one(
                model,
                mel,
                options,
                keyword_tokens=token_ids,
                beta=float(beta),
                image_path=image_path,
                use_images=use_images,
            )
            predictions.append(prediction_row(row, index, result.text, beta=float(beta), keywords=keywords, mode=args.keyword_mode))
            if index == 1 or index == len(rows) or index % max(1, args.log_every) == 0:
                print(f"[A14] beta={beta} row={index}/{len(rows)} elapsed={time.time() - started:.1f}s")
        beta_key = str(beta).replace(".", "p")
        write_jsonl(output_dir / f"predictions_beta_{beta_key}.jsonl", predictions)
        all_metrics[str(beta)] = prediction_metrics(predictions)

    summary = {
        "manifest_path": str(manifest_path),
        "image_semantics_jsonl": str(resolve_cross_platform_path(args.image_semantics_jsonl)),
        "keyword_mode": args.keyword_mode,
        "rows": len(rows),
        "beam_size": int(args.beam_size),
        "model_source": model_source,
        "metrics_by_beta": all_metrics,
        "note": "Select beta on validation only; evaluate selected beta once on test.",
    }
    write_json(output_dir / "metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
