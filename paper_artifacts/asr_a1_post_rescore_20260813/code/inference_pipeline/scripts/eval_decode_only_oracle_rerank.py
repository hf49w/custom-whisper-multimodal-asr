"""Decode-only ASR evaluation with optional n-best oracle and CLIP reranking.

This script deliberately avoids ``model.transcribe()``.  Every sample is
evaluated as one 30-second padded/trimmed window with ``model.decode()``.  This
matches the decode path used by ``eval_clip_rerank.py`` and makes normal,
oracle, and reranked metrics directly comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import custom_whisper
from eval_visspeech_custom_whisper_fuser import rebuild_model
from visspeech_custom_whisper_utils import (
    compute_cer,
    compute_wer,
    load_manifest,
    normalize_eval_text,
    resolve_cross_platform_path,
    summarize_predictions,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint-path", default="")
    source.add_argument("--whisper-model", default="")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--n-best", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--disable-images", action="store_true")
    parser.add_argument("--skip-if-exists", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.set_defaults(no_download=True)
    parser.add_argument("--no-download", dest="no_download", action="store_true")
    parser.add_argument("--allow-download", dest="no_download", action="store_false")
    parser.add_argument("--clip-model-name", default="")
    parser.add_argument("--visual-model-name", default="")
    parser.add_argument("--clip-rerank-lambda", type=float, nargs="*", default=[])
    parser.add_argument("--prompt-jsonl", default="")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--prompt-key", choices=["image_id", "image_path", "key"], default="image_id")
    parser.add_argument("--prompt-mode", choices=["prompt", "prefix"], default="prompt")
    parser.add_argument("--prompt-template", default="{prompt}")
    parser.add_argument("--shuffle-prompts", action="store_true")
    return parser.parse_args()


class ClipCandidateScorer:
    def __init__(self, model_name: str, device: torch.device, no_download: bool):
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError("CLIP reranking requires transformers with CLIPModel support") from exc

        self.processor = CLIPProcessor.from_pretrained(
            model_name, local_files_only=no_download
        )
        self.model = CLIPModel.from_pretrained(
            model_name, local_files_only=no_download
        ).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def score(self, image_path: str, texts: Sequence[str]) -> List[float]:
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


def unique_candidates(
    texts: Sequence[str], scores: Sequence[float], limit: int
) -> List[Tuple[str, float]]:
    candidates: List[Tuple[str, float]] = []
    seen = set()
    for text, score in sorted(zip(texts, scores), key=lambda item: item[1], reverse=True):
        normalized = str(text).strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append((normalized, float(score)))
        if len(candidates) >= limit:
            break
    return candidates


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


def oracle_choice(ref_text: str, candidates: Sequence[Tuple[str, float]]) -> Dict[str, Any]:
    ref_norm = normalize_eval_text(ref_text)
    ref_words = ref_norm.split()
    ref_chars = list(ref_norm)
    best_index = 0
    best_key = None
    best_word_edits = 0
    best_char_edits = 0
    for index, (text, _score) in enumerate(candidates):
        hyp_norm = normalize_eval_text(text)
        word_edits = edit_distance(ref_words, hyp_norm.split())
        char_edits = edit_distance(ref_chars, list(hyp_norm))
        key = (word_edits, char_edits, index)
        if best_key is None or key < best_key:
            best_key = key
            best_index = index
            best_word_edits = word_edits
            best_char_edits = char_edits
    selected_text, selected_score = candidates[best_index]
    return {
        "index": best_index,
        "text": selected_text,
        "asr_logprob": selected_score,
        "word_edits": best_word_edits,
        "char_edits": best_char_edits,
    }


def build_prediction(row: Dict[str, Any], index: int, pred_text: str) -> Dict[str, Any]:
    ref_text = str(row.get("annotation", ""))
    wav_path = str(row["wav_path"])
    image_path = str(row.get("image_path", ""))
    return {
        "key": str(row.get("key") or Path(wav_path).stem or index),
        "wav_path": wav_path,
        "image_path": image_path,
        "ref_text": ref_text,
        "pred_text": str(pred_text).strip(),
        "norm_ref_text": normalize_eval_text(ref_text),
        "norm_pred_text": normalize_eval_text(pred_text),
    }


def model_needs_images(model: Any, *, is_checkpoint: bool, disable_images: bool) -> bool:
    if not is_checkpoint or disable_images:
        return False
    visual_config = getattr(model, "visual_config", {}) or {}
    visual_name = str(visual_config.get("visual_encoder", "")).lower()
    adapter_name = str(getattr(model, "decoder_prompt_adapter_name", "")).lower()
    if visual_name in {"", "none", "no_visual", "novisualencoder"}:
        return False
    if adapter_name == "blank_prefix":
        return False
    return hasattr(model, "use_visual_context")


def load_eval_model(args: argparse.Namespace, device: torch.device):
    if args.checkpoint_path:
        checkpoint_path = resolve_cross_platform_path(args.checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint = dict(checkpoint)
        checkpoint["train_config"] = dict(checkpoint["train_config"])
        base_whisper_model = str(getattr(args, "base_whisper_model", "") or "").strip()
        if base_whisper_model:
            checkpoint["train_config"]["whisper_model"] = base_whisper_model
        if args.visual_model_name:
            checkpoint["train_config"]["clip_model_name"] = args.visual_model_name
        blip2_model_name = str(getattr(args, "blip2_model_name", "") or "").strip()
        if blip2_model_name:
            checkpoint["train_config"]["blip2_model_name"] = blip2_model_name
        model = rebuild_model(checkpoint, device=device, no_download=args.no_download).eval()
        return model, str(checkpoint_path), True

    whisper_model = resolve_cross_platform_path(args.whisper_model)
    model = custom_whisper.load_model(str(whisper_model), device=device).eval()
    return model, str(whisper_model), False


def load_prompt_map(args: argparse.Namespace, rows: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    if not args.prompt_jsonl:
        return {}
    prompt_path = resolve_cross_platform_path(args.prompt_jsonl)
    raw_prompts: Dict[str, str] = {}
    with prompt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            key_value = str(item.get(args.prompt_key, ""))
            if not key_value and args.prompt_key == "image_path":
                key_value = str(item.get("image_filename", ""))
            prompt_value = str(item.get(args.prompt_field, "")).strip()
            if key_value and prompt_value:
                raw_prompts[key_value] = prompt_value

    def row_key(row: Dict[str, Any]) -> str:
        if args.prompt_key == "image_path":
            return str(row.get("image_path", ""))
        return str(row.get(args.prompt_key, ""))

    if not args.shuffle_prompts:
        return raw_prompts

    ordered_keys = []
    seen = set()
    for row in rows:
        key = row_key(row)
        if key and key in raw_prompts and key not in seen:
            ordered_keys.append(key)
            seen.add(key)
    if len(ordered_keys) <= 1:
        return raw_prompts
    shifted_values = [raw_prompts[key] for key in ordered_keys[1:]] + [raw_prompts[ordered_keys[0]]]
    shuffled = dict(raw_prompts)
    for key, value in zip(ordered_keys, shifted_values):
        shuffled[key] = value
    return shuffled


def row_prompt(args: argparse.Namespace, prompt_map: Dict[str, str], row: Dict[str, Any]) -> str:
    if not prompt_map:
        return ""
    key = str(row.get(args.prompt_key, ""))
    if args.prompt_key == "image_path":
        key = str(row.get("image_path", ""))
    prompt_value = prompt_map.get(key, "")
    if not prompt_value:
        return ""
    return args.prompt_template.format(prompt=prompt_value).strip()


def main() -> None:
    args = parse_args()
    output_root = resolve_cross_platform_path(args.output_root)
    metrics_path = output_root / "metrics.json"
    if args.skip_if_exists and metrics_path.is_file():
        print(f"[SKIP] metrics exists: {metrics_path}")
        return
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = resolve_cross_platform_path(args.manifest_path)
    rows = load_manifest(manifest_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError(f"No rows loaded from {manifest_path}")
    prompt_map = load_prompt_map(args, rows)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, model_source, is_checkpoint = load_eval_model(args, device)
    n_best = args.n_best if args.n_best > 0 else args.beam_size
    n_best = max(1, min(n_best, args.beam_size))

    clip_scorer: Optional[ClipCandidateScorer] = None
    if args.clip_rerank_lambda:
        if not args.clip_model_name:
            raise ValueError("--clip-model-name is required when --clip-rerank-lambda is set")
        clip_scorer = ClipCandidateScorer(args.clip_model_name, device, args.no_download)

    base_options = {
        "language": "en",
        "task": "transcribe",
        "beam_size": args.beam_size,
        "fp16": device.type == "cuda",
        "without_timestamps": True,
    }

    before_predictions: List[Dict[str, Any]] = []
    oracle_predictions: List[Dict[str, Any]] = []
    rerank_predictions: Dict[float, List[Dict[str, Any]]] = {
        value: [] for value in args.clip_rerank_lambda
    }
    nbest_rows: List[Dict[str, Any]] = []
    started = time.time()

    for index, row in enumerate(rows, start=1):
        audio = custom_whisper.pad_or_trim(custom_whisper.load_audio(str(row["wav_path"])))
        mel = custom_whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(device)

        decode_kwargs: Dict[str, Any] = {}
        use_images = model_needs_images(model, is_checkpoint=is_checkpoint, disable_images=args.disable_images)
        if use_images:
            decode_kwargs["image"] = str(row["image_path"])
        active_prompt = row_prompt(args, prompt_map, row)
        option_kwargs = dict(base_options)
        if active_prompt:
            option_kwargs[args.prompt_mode] = active_prompt
        options = custom_whisper.DecodingOptions(**option_kwargs)
        result = model.decode(mel, options, **decode_kwargs)

        raw_texts = result.nbest_texts or [result.text]
        raw_scores = result.nbest_avg_logprobs or [result.avg_logprob]
        candidates = unique_candidates(raw_texts, raw_scores, n_best)
        if not candidates:
            candidates = [(str(result.text), float(result.avg_logprob))]

        common = build_prediction(row, index, result.text)
        if active_prompt:
            common["decode_prompt"] = active_prompt
            common["decode_prompt_mode"] = args.prompt_mode
        before_predictions.append(common)
        nbest_rows.append(
            {
                **{key: common[key] for key in ("key", "wav_path", "image_path", "ref_text")},
                "decode_prompt": active_prompt,
                "decode_prompt_mode": args.prompt_mode if active_prompt else "",
                "beam_size": args.beam_size,
                "n_best": n_best,
                "candidates": [
                    {
                        "rank": rank,
                        "text": text,
                        "asr_logprob": score,
                        "norm_text": normalize_eval_text(text),
                    }
                    for rank, (text, score) in enumerate(candidates)
                ],
            }
        )

        oracle = oracle_choice(str(row.get("annotation", "")), candidates)
        oracle_predictions.append(
            {
                **build_prediction(row, index, oracle["text"]),
                "oracle_index": oracle["index"],
                "oracle_asr_logprob": oracle["asr_logprob"],
                "oracle_word_edits": oracle["word_edits"],
                "oracle_char_edits": oracle["char_edits"],
            }
        )

        if clip_scorer is not None:
            candidate_texts = [text for text, _score in candidates]
            asr_scores = [score for _text, score in candidates]
            clip_scores = clip_scorer.score(str(row["image_path"]), candidate_texts)
            for value in args.clip_rerank_lambda:
                final_scores = [asr + float(value) * clip for asr, clip in zip(asr_scores, clip_scores)]
                selected = max(range(len(final_scores)), key=final_scores.__getitem__)
                pred = build_prediction(row, index, candidate_texts[selected])
                pred["selected_index"] = selected
                pred["candidates"] = [
                    {
                        "rank": rank,
                        "text": text,
                        "asr_logprob": asr,
                        "clip_score": clip,
                        "final_score": final,
                    }
                    for rank, (text, asr, clip, final) in enumerate(
                        zip(candidate_texts, asr_scores, clip_scores, final_scores)
                    )
                ]
                rerank_predictions[value].append(pred)

        if index == 1 or index == len(rows) or index % max(1, args.log_every) == 0:
            print(f"[DECODE] row={index}/{len(rows)} candidates={len(candidates)}")

    before_metrics = summarize_predictions(before_predictions)
    oracle_metrics = summarize_predictions(oracle_predictions)
    summary: Dict[str, Any] = {
        "model_source": model_source,
        "checkpoint_path": str(resolve_cross_platform_path(args.checkpoint_path)) if args.checkpoint_path else "",
        "whisper_model": str(resolve_cross_platform_path(args.whisper_model)) if args.whisper_model else "",
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "rows": len(rows),
        "beam_size": args.beam_size,
        "n_best": n_best,
        "use_images": model_needs_images(model, is_checkpoint=is_checkpoint, disable_images=args.disable_images),
        "decode_path": "pad_or_trim+model.decode",
        "prompt_jsonl": str(resolve_cross_platform_path(args.prompt_jsonl)) if args.prompt_jsonl else "",
        "prompt_field": args.prompt_field if args.prompt_jsonl else "",
        "prompt_mode": args.prompt_mode if args.prompt_jsonl else "",
        "prompt_template": args.prompt_template if args.prompt_jsonl else "",
        "shuffle_prompts": bool(args.shuffle_prompts),
        "before": before_metrics,
        "oracle": oracle_metrics,
        "seconds": time.time() - started,
        "lambda": {},
    }

    write_jsonl(output_root / "predictions_before.jsonl", before_predictions)
    write_jsonl(output_root / "predictions_oracle.jsonl", oracle_predictions)
    write_jsonl(output_root / "nbest.jsonl", nbest_rows)
    for value, predictions in rerank_predictions.items():
        metrics = summarize_predictions(predictions)
        summary["lambda"][str(value)] = metrics
        write_jsonl(output_root / f"predictions_clip_lambda_{value:g}.jsonl", predictions)

    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
