"""Evaluate an unmodified Whisper checkpoint on a prepared Flickr8K manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import custom_whisper
from visspeech_custom_whisper_utils import (
    load_manifest,
    resolve_cross_platform_path,
    summarize_predictions,
    transcribe_manifest_rows,
    write_jsonl,
)


def main() -> None:
    """Run the pure Whisper baseline and persist predictions and metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whisper-model", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = custom_whisper.load_model(args.whisper_model, device=device).eval()
    rows = load_manifest(resolve_cross_platform_path(args.manifest_path))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    output_root = resolve_cross_platform_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    predictions = transcribe_manifest_rows(
        model,
        rows,
        use_images=False,
        fp16=device.type == "cuda",
        transcribe_kwargs={"beam_size": args.beam_size},
    )
    metrics = summarize_predictions(predictions)
    write_jsonl(output_root / "predictions.jsonl", predictions)
    (output_root / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
