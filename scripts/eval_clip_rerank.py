"""Evaluate image-aware CLIP reranking over Whisper beam candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
    load_manifest,
    resolve_cross_platform_path,
    summarize_predictions,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    """Parse reranking and local-model options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-clip-rerank", action="store_true")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--whisper-model", default="")
    parser.add_argument("--visual-model-name", default="")
    parser.add_argument("--clip-model-name", required=True)
    parser.add_argument("--clip-rerank-lambda", type=float, nargs="+", default=[0.1])
    parser.add_argument("--rerank-n-best", type=int, default=5)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--device", default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.set_defaults(no_download=True)
    parser.add_argument("--no-download", dest="no_download", action="store_true")
    parser.add_argument("--allow-download", dest="no_download", action="store_false")
    return parser.parse_args()


class ClipCandidateScorer:
    """Score image/text pairs with a local or explicitly allowed CLIP model."""

    def __init__(self, model_name: str, device: torch.device, no_download: bool):
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError("CLIP reranking requires transformers with CLIPModel support") from exc
        try:
            self.processor = CLIPProcessor.from_pretrained(
                model_name, local_files_only=no_download
            )
            self.model = CLIPModel.from_pretrained(
                model_name, local_files_only=no_download
            ).to(device)
        except OSError as exc:
            raise FileNotFoundError(
                "CLIP reranker weights are unavailable locally. Pass a local directory "
                "with --clip-model-name or explicitly use --allow-download."
            ) from exc
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def score(self, image_path: str, texts: Sequence[str]) -> List[float]:
        """Return cosine similarities for one image and candidate texts."""
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


def unique_candidates(texts: Sequence[str], scores: Sequence[float], limit: int):
    """Deduplicate candidates while preserving descending ASR score order."""
    candidates = []
    seen = set()
    for text, score in sorted(zip(texts, scores), key=lambda item: item[1], reverse=True):
        normalized = text.strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append((normalized, float(score)))
        if len(candidates) >= limit:
            break
    return candidates


def main() -> None:
    """Decode the manifest, rerank candidates, and write metrics."""
    args = parse_args()
    if not args.enable_clip_rerank:
        raise ValueError("Pass --enable-clip-rerank to run this evaluation")
    if args.rerank_n_best <= 0 or args.beam_size < args.rerank_n_best:
        raise ValueError("beam-size must be >= rerank-n-best > 0")

    checkpoint_path = resolve_cross_platform_path(args.checkpoint_path)
    manifest_path = resolve_cross_platform_path(args.manifest_path)
    output_root = resolve_cross_platform_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint = dict(checkpoint)
    checkpoint["train_config"] = dict(checkpoint["train_config"])
    if args.whisper_model:
        checkpoint["train_config"]["whisper_model"] = args.whisper_model
    if args.visual_model_name:
        checkpoint["train_config"]["clip_model_name"] = args.visual_model_name
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = rebuild_model(checkpoint, device=device, no_download=args.no_download).eval()
    scorer = ClipCandidateScorer(args.clip_model_name, device, args.no_download)
    rows = load_manifest(manifest_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    options = custom_whisper.DecodingOptions(
        language="en",
        task="transcribe",
        beam_size=args.beam_size,
        fp16=device.type == "cuda",
        without_timestamps=True,
    )
    outputs_by_lambda: Dict[float, List[Dict[str, Any]]] = {
        value: [] for value in args.clip_rerank_lambda
    }
    before_predictions: List[Dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        audio = custom_whisper.pad_or_trim(custom_whisper.load_audio(str(row["wav_path"])))
        mel = custom_whisper.log_mel_spectrogram(
            audio, n_mels=model.dims.n_mels
        ).to(device)
        result = model.decode(mel, options, image=str(row["image_path"]))
        candidates = unique_candidates(
            result.nbest_texts or [result.text],
            result.nbest_avg_logprobs or [result.avg_logprob],
            args.rerank_n_best,
        )
        candidate_texts = [item[0] for item in candidates]
        asr_scores = [item[1] for item in candidates]
        clip_scores = scorer.score(str(row["image_path"]), candidate_texts)
        common = {
            "key": str(row.get("key", index)),
            "wav_path": str(row["wav_path"]),
            "image_path": str(row["image_path"]),
            "ref_text": str(row.get("annotation", "")),
        }
        before_predictions.append(dict(common, pred_text=result.text))
        for value in args.clip_rerank_lambda:
            final_scores = [
                asr + float(value) * clip
                for asr, clip in zip(asr_scores, clip_scores)
            ]
            selected = max(range(len(final_scores)), key=final_scores.__getitem__)
            outputs_by_lambda[value].append(
                dict(
                    common,
                    pred_text=candidate_texts[selected],
                    candidates=[
                        {
                            "text": text,
                            "asr_logprob": asr,
                            "clip_score": clip,
                            "final_score": final,
                        }
                        for text, asr, clip, final in zip(
                            candidate_texts, asr_scores, clip_scores, final_scores
                        )
                    ],
                )
            )
        print(f"[RERANK] row={index}/{len(rows)} candidates={len(candidates)}")

    before_metrics = summarize_predictions(before_predictions)
    summary: Dict[str, Any] = {"before": before_metrics, "lambda": {}}
    write_jsonl(output_root / "predictions_before.jsonl", before_predictions)
    for value, predictions in outputs_by_lambda.items():
        metrics = summarize_predictions(predictions)
        summary["lambda"][str(value)] = metrics
        write_jsonl(output_root / f"predictions_lambda_{value:g}.jsonl", predictions)
    (output_root / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
