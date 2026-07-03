"""Dump decode-only n-best candidates for A9 candidate reranking.

The decode path intentionally matches ``eval_decode_only_oracle_rerank.py``:
one padded/trimmed 30-second window per row, ``log_mel_spectrogram`` followed by
``model.decode()`` with ``language='en'``, ``task='transcribe'`` and
``without_timestamps=True``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import custom_whisper
from a9_candidate_utils import normalize_eval_text
from custom_whisper.utils import compression_ratio
from eval_decode_only_oracle_rerank import load_eval_model, model_needs_images
from visspeech_custom_whisper_utils import (
    build_tokenizer_and_prefix,
    load_manifest,
    resolve_cross_platform_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint-path", default="")
    source.add_argument("--whisper-model", default="")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--beam-size", type=int, required=True, choices=[10, 20, 30, 50])
    parser.add_argument("--n-best", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--disable-images", action="store_true")
    parser.add_argument("--skip-if-exists", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.set_defaults(no_download=True)
    parser.add_argument("--no-download", dest="no_download", action="store_true")
    parser.add_argument("--allow-download", dest="no_download", action="store_false")
    parser.add_argument("--visual-model-name", default="")
    return parser.parse_args()


def sample_id_for_row(row: Dict[str, Any], index: int) -> str:
    wav_path = str(row.get("wav_path", ""))
    return str(row.get("key") or row.get("utt_id") or Path(wav_path).stem or index)


def unique_candidates(
    texts: Sequence[str],
    scores: Sequence[float],
    tokens: Sequence[Sequence[int]],
    limit: int,
) -> List[Tuple[str, float, List[int]]]:
    packed = []
    for text, score, token_ids in zip(texts, scores, tokens):
        packed.append((str(text).strip(), float(score), [int(token) for token in token_ids]))
    packed.sort(key=lambda item: item[1], reverse=True)

    candidates: List[Tuple[str, float, List[int]]] = []
    seen = set()
    for text, score, token_ids in packed:
        key = normalize_eval_text(text)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((text, score, token_ids))
        if len(candidates) >= limit:
            break
    return candidates


@torch.no_grad()
def teacher_forced_token_logprobs(
    model: Any,
    audio_features: torch.Tensor,
    *,
    initial_tokens: Sequence[int],
    candidate_tokens: Sequence[int],
    eot_token: int,
    device: torch.device,
    image_path: str = "",
    use_images: bool = False,
) -> List[float]:
    full_tokens = list(initial_tokens) + [int(token) for token in candidate_tokens] + [int(eot_token)]
    if len(full_tokens) < 2:
        return []
    input_tokens = torch.tensor(full_tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
    labels = torch.tensor(full_tokens[1:], dtype=torch.long, device=device).unsqueeze(0)
    features = audio_features
    if features.dim() == 2:
        features = features.unsqueeze(0)
    features = features.to(device)

    if use_images and hasattr(model, "use_visual_context"):
        with model.use_visual_context(image=image_path):
            logits = model.logits(input_tokens, features)
    else:
        logits = model.logits(input_tokens, features)
    logprobs = F.log_softmax(logits.float(), dim=-1)
    gathered = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(0).squeeze(-1)
    start = max(0, len(initial_tokens) - 1)
    end = start + len(candidate_tokens) + 1
    return [float(value) for value in gathered[start:end].detach().cpu().tolist()]


def candidate_record(
    *,
    text: str,
    normalized_text: str,
    beam_rank: int,
    mean_logprob: float,
    token_ids: Sequence[int],
    token_logprobs: Sequence[float],
) -> Dict[str, Any]:
    token_count = len(token_ids)
    words = normalized_text.split()
    sum_logprob = float(mean_logprob) * float(token_count + 1)
    avg_token_logprob = float(sum(token_logprobs) / len(token_logprobs)) if token_logprobs else float(mean_logprob)
    min_token_logprob = float(min(token_logprobs)) if token_logprobs else float(mean_logprob)
    return {
        "text": text,
        "normalized_text": normalized_text,
        "beam_rank": int(beam_rank),
        "sum_logprob": sum_logprob,
        "mean_logprob": float(mean_logprob),
        "asr_mean_logprob": float(mean_logprob),
        "avg_token_logprob": avg_token_logprob,
        "min_token_logprob": min_token_logprob,
        "token_count": int(token_count),
        "word_count": int(len(words)),
        "char_count": int(len(normalized_text)),
        "compression_ratio": float(compression_ratio(text or " ")),
    }


def main() -> None:
    args = parse_args()
    output_jsonl = resolve_cross_platform_path(args.output_jsonl)
    if args.skip_if_exists and output_jsonl.is_file() and output_jsonl.stat().st_size > 0:
        print(f"[SKIP] output exists: {output_jsonl}")
        return
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = resolve_cross_platform_path(args.manifest_path)
    rows = load_manifest(manifest_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError(f"No rows loaded from {manifest_path}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, model_source, is_checkpoint = load_eval_model(args, device)
    tokenizer, initial_tokens = build_tokenizer_and_prefix(model)
    n_best = args.n_best if args.n_best > 0 else args.beam_size
    n_best = max(1, min(n_best, args.beam_size))
    use_images = model_needs_images(model, is_checkpoint=is_checkpoint, disable_images=args.disable_images)

    options = custom_whisper.DecodingOptions(
        language="en",
        task="transcribe",
        beam_size=args.beam_size,
        fp16=device.type == "cuda",
        without_timestamps=True,
    )

    started = time.time()
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            audio_path = str(row["wav_path"])
            image_path = str(row.get("image_path", ""))
            audio = custom_whisper.pad_or_trim(custom_whisper.load_audio(audio_path))
            mel = custom_whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(device)
            decode_kwargs: Dict[str, Any] = {}
            if use_images:
                decode_kwargs["image"] = image_path
            result = model.decode(mel, options, **decode_kwargs)

            raw_texts = result.nbest_texts or [result.text]
            raw_scores = result.nbest_avg_logprobs or [result.avg_logprob]
            raw_tokens = result.nbest_tokens or [result.tokens]
            candidates = unique_candidates(raw_texts, raw_scores, raw_tokens, n_best)
            if not candidates:
                candidates = [(str(result.text), float(result.avg_logprob), list(result.tokens))]

            candidate_rows: List[Dict[str, Any]] = []
            for rank, (text, score, token_ids) in enumerate(candidates):
                token_logprobs = teacher_forced_token_logprobs(
                    model,
                    result.audio_features,
                    initial_tokens=initial_tokens,
                    candidate_tokens=token_ids,
                    eot_token=tokenizer.eot,
                    device=device,
                    image_path=image_path,
                    use_images=use_images,
                )
                normalized_text = normalize_eval_text(text)
                candidate_rows.append(
                    candidate_record(
                        text=text,
                        normalized_text=normalized_text,
                        beam_rank=rank,
                        mean_logprob=score,
                        token_ids=token_ids,
                        token_logprobs=token_logprobs,
                    )
                )

            ref = str(row.get("annotation", row.get("text", "")))
            sample = {
                "sample_id": sample_id_for_row(row, index),
                "key": str(row.get("key") or sample_id_for_row(row, index)),
                "audio_path": audio_path,
                "wav_path": audio_path,
                "image_path": image_path,
                "reference": ref,
                "normalized_reference": normalize_eval_text(ref),
                "beam_size": int(args.beam_size),
                "n_best": int(n_best),
                "decode_path": "pad_or_trim+model.decode",
                "model_source": model_source,
                "candidates": candidate_rows,
            }
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

            if index == 1 or index == len(rows) or index % max(1, args.log_every) == 0:
                elapsed = time.time() - started
                print(
                    f"[DUMP] row={index}/{len(rows)} candidates={len(candidate_rows)} "
                    f"elapsed={elapsed:.1f}s"
                )


if __name__ == "__main__":
    main()
