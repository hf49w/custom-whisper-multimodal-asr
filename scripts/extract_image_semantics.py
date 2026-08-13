"""Extract offline image semantics for A12/A13 reranking.

The generated captions/tags are never fed into Whisper prompts by this script.
They are saved as side-channel reranking features for n-best candidates.

If a local BLIP/BLIP-2 model is unavailable, pass --existing-semantics-jsonl to
normalize an already generated captions/tags file into the schema used by the
A12/A13 scripts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a9_candidate_utils import normalize_eval_text, read_jsonl, sample_id
from visspeech_custom_whisper_utils import load_manifest, resolve_cross_platform_path


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

SCENE_WORDS = {
    "beach",
    "city",
    "field",
    "forest",
    "grass",
    "mountain",
    "park",
    "road",
    "sidewalk",
    "snow",
    "street",
    "track",
    "trail",
    "water",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--existing-semantics-jsonl", default="")
    parser.add_argument("--model-type", choices=["auto", "blip", "blip2", "none"], default="auto")
    parser.add_argument("--model-name", default="", help="Local BLIP/BLIP-2 model directory.")
    parser.add_argument("--device", default="")
    parser.add_argument("--num-captions", type=int, default=3)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.set_defaults(no_download=True)
    parser.add_argument("--no-download", dest="no_download", action="store_true")
    parser.add_argument("--allow-download", dest="no_download", action="store_false")
    return parser.parse_args()


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_sample_id(row: Dict[str, Any], index: int = 0) -> str:
    if sample_id(row):
        return sample_id(row)
    wav_path = str(row.get("wav_path") or row.get("audio_path") or "")
    return str(row.get("key") or row.get("utt_id") or Path(wav_path).stem or index)


def as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        values: List[str] = []
        for item in value.values():
            values.extend(as_list(item))
        return values
    if isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            values.extend(as_list(item))
        return values
    return [str(value)]


def unique_clean_words(values: Sequence[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        for token in normalize_eval_text(str(value)).split():
            if len(token) <= 1 or token in STOPWORDS:
                continue
            if token not in seen:
                seen.add(token)
                output.append(token)
    return output


def categorize_tags(words: Sequence[str]) -> Dict[str, List[str]]:
    verbs: List[str] = []
    adjectives: List[str] = []
    scenes: List[str] = []
    objects: List[str] = []
    for word in words:
        if word in COMMON_VERBS or word.endswith("ing"):
            verbs.append(word)
        elif word in COMMON_ADJECTIVES:
            adjectives.append(word)
        elif word in SCENE_WORDS:
            scenes.append(word)
        else:
            objects.append(word)
    return {
        "objects": objects,
        "actions": verbs,
        "attributes": adjectives,
        "scenes": scenes,
        "all": list(words),
    }


def semantics_from_existing(existing_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mapped: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(existing_rows, start=1):
        sid = row_sample_id(row, index)
        image_path = str(row.get("image_path") or "")
        captions = as_list(row.get("captions") or row.get("caption"))
        raw_tags = as_list(row.get("tags"))
        if not raw_tags:
            raw_tags = unique_clean_words(captions)
        words = unique_clean_words(raw_tags + captions)
        tags = categorize_tags(words)
        normalized = {
            "sample_id": sid,
            "image_path": image_path,
            "captions": captions,
            "tags": tags,
            "scores": row.get("scores", {}),
            "source": row.get("source", "existing"),
        }
        mapped[sid] = normalized
        if image_path:
            mapped[image_path] = normalized
    return mapped


class LocalCaptioner:
    def __init__(self, *, model_type: str, model_name: str, device: torch.device, no_download: bool):
        if not model_name:
            raise ValueError("--model-name is required unless --existing-semantics-jsonl is used with --model-type none")
        try:
            from transformers import (
                Blip2ForConditionalGeneration,
                Blip2Processor,
                BlipForConditionalGeneration,
                BlipProcessor,
            )
        except ImportError as exc:
            raise ImportError("Image captioning requires transformers with BLIP/BLIP-2 support") from exc

        model_name = str(resolve_cross_platform_path(model_name))
        if model_type == "auto":
            lowered = model_name.lower()
            model_type = "blip2" if "blip2" in lowered else "blip"
        self.model_type = model_type
        self.device = device
        if model_type == "blip2":
            self.processor = Blip2Processor.from_pretrained(model_name, local_files_only=no_download)
            self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, local_files_only=no_download).to(device)
        elif model_type == "blip":
            self.processor = BlipProcessor.from_pretrained(model_name, local_files_only=no_download)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name, local_files_only=no_download).to(device)
        else:
            raise ValueError(f"Unsupported model_type for caption generation: {model_type}")
        self.model.eval()

    @torch.no_grad()
    def captions(self, image_path: str, *, num_captions: int, num_beams: int, max_new_tokens: int) -> List[str]:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        generated = self.model.generate(
            **inputs,
            num_beams=max(1, int(num_beams)),
            num_return_sequences=max(1, int(num_captions)),
            max_new_tokens=max(1, int(max_new_tokens)),
        )
        captions = self.processor.batch_decode(generated, skip_special_tokens=True)
        cleaned: List[str] = []
        seen = set()
        for caption in captions:
            text = str(caption).strip()
            key = normalize_eval_text(text)
            if text and key not in seen:
                seen.add(key)
                cleaned.append(text)
        return cleaned


def main() -> None:
    args = parse_args()
    manifest_path = resolve_cross_platform_path(args.manifest)
    output_jsonl = resolve_cross_platform_path(args.output_jsonl)
    rows = load_manifest(manifest_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError(f"No manifest rows loaded from {manifest_path}")

    existing: Dict[str, Dict[str, Any]] = {}
    if args.existing_semantics_jsonl:
        existing = semantics_from_existing(read_jsonl(resolve_cross_platform_path(args.existing_semantics_jsonl)))

    captioner = None
    if args.model_type != "none" and args.model_name:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        captioner = LocalCaptioner(
            model_type=args.model_type,
            model_name=args.model_name,
            device=device,
            no_download=args.no_download,
        )
    elif not existing:
        raise ValueError("Provide --model-name or --existing-semantics-jsonl")

    started = time.time()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            sid = row_sample_id(row, index)
            image_path = str(row.get("image_path") or row.get("image") or "")
            existing_row = existing.get(sid) or existing.get(image_path)
            if existing_row and captioner is None:
                out = dict(existing_row)
                out["sample_id"] = sid
                out["image_path"] = image_path or str(existing_row.get("image_path", ""))
            else:
                captions = captioner.captions(
                    image_path,
                    num_captions=args.num_captions,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                ) if captioner is not None else as_list(existing_row.get("captions"))
                words = unique_clean_words(captions + as_list((existing_row or {}).get("tags")))
                out = {
                    "sample_id": sid,
                    "image_path": image_path,
                    "captions": captions,
                    "tags": categorize_tags(words),
                    "scores": {},
                    "source": "generated" if captioner is not None else "existing",
                    "model_type": args.model_type,
                    "model_name": str(resolve_cross_platform_path(args.model_name)) if args.model_name else "",
                }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            if index == 1 or index == len(rows) or index % max(1, args.log_every) == 0:
                print(f"[SEMANTICS] row={index}/{len(rows)} elapsed={time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
