"""Generate per-image captions and simple keyword tags for prompt experiments."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visspeech_custom_whisper_utils import load_manifest, resolve_cross_platform_path


STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "on", "in", "at", "to", "with", "for",
    "from", "by", "is", "are", "be", "being", "been", "this", "that", "these",
    "those", "there", "here", "photo", "image", "picture", "showing", "shows",
    "standing", "sitting", "looking", "wearing", "holding", "front", "background",
    "near", "next", "while", "into", "over", "under", "up", "down", "out",
}
WORD_RE = re.compile(r"[a-z][a-z'-]*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--generation-prompt", default="")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def read_existing(path: Path) -> Dict[str, Dict]:
    if not path.is_file():
        return {}
    rows: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            image_id = str(item.get("image_id", ""))
            if image_id:
                rows[image_id] = item
    return rows


def append_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def unique_images(manifest_path: Path) -> List[Dict]:
    rows = load_manifest(manifest_path)
    by_id: Dict[str, Dict] = {}
    for row in rows:
        image_id = str(row.get("image_id") or Path(str(row["image_path"])).stem)
        if image_id not in by_id:
            by_id[image_id] = {
                "image_id": image_id,
                "image_path": str(row["image_path"]),
                "image_filename": str(row.get("image_filename") or Path(str(row["image_path"])).name),
            }
    return list(by_id.values())


def simple_tags(caption: str, limit: int = 8) -> List[str]:
    tags: List[str] = []
    seen = set()
    for word in WORD_RE.findall(caption.lower()):
        word = word.strip("-'")
        if len(word) < 3 or word in STOPWORDS or word in seen:
            continue
        seen.add(word)
        tags.append(word)
        if len(tags) >= limit:
            break
    return tags


def main() -> None:
    args = parse_args()
    manifest_path = resolve_cross_platform_path(args.manifest_path)
    output_path = resolve_cross_platform_path(args.output_jsonl)
    model_name = str(resolve_cross_platform_path(args.model_name))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    try:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
    except ImportError as exc:
        raise ImportError("BLIP2 caption generation requires transformers") from exc

    images = unique_images(manifest_path)
    if args.max_images > 0:
        images = images[: args.max_images]
    existing = read_existing(output_path)

    processor = Blip2Processor.from_pretrained(model_name, local_files_only=True)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    model.eval()

    done = len([item for item in images if item["image_id"] in existing])
    print(f"[INFO] images={len(images)} existing={done} remaining={len(images)-done}")
    for index, item in enumerate(images, start=1):
        image_id = item["image_id"]
        if image_id in existing:
            continue
        image = Image.open(item["image_path"]).convert("RGB")
        if args.generation_prompt:
            inputs = processor(images=image, text=args.generation_prompt, return_tensors="pt")
        else:
            inputs = processor(images=image, return_tensors="pt")
        inputs = {
            key: (value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device))
            for key, value in inputs.items()
        }
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        caption = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        tags = simple_tags(caption)
        record = {
            **item,
            "caption": caption,
            "tags": tags,
            "caption_prompt": f"Context: {caption}.",
            "tags_prompt": f"Context: {', '.join(tags)}." if tags else "",
        }
        append_jsonl(output_path, [record])
        if index == 1 or index == len(images) or index % max(1, args.log_every) == 0:
            print(f"[CAPTION] image={index}/{len(images)} generated={image_id} caption={caption}")

    print(f"[DONE] output={output_path}")


if __name__ == "__main__":
    main()
