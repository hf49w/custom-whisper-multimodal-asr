"""Build a fixed train-subset manifest for rerank development selection.

This script samples rows from an existing train manifest. It does not train a
model and does not touch checkpoints. The resulting manifest can be passed to
dump_nbest_candidates.py, then the dumped n-best JSONL can be combined with val
via rerank_nbest_candidates.py --dev-nbest.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visspeech_custom_whisper_utils import resolve_cross_platform_path, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--sample-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle-output",
        action="store_true",
        help="Write sampled rows in random order. Default preserves original manifest order.",
    )
    parser.add_argument("--summary-json", default="")
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    train_manifest = resolve_cross_platform_path(args.train_manifest)
    output_manifest = resolve_cross_platform_path(args.output_manifest)
    rows = read_jsonl(train_manifest)
    if not rows:
        raise ValueError(f"No rows loaded from {train_manifest}")
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    rng = random.Random(int(args.seed))
    sample_size = min(int(args.sample_size), len(rows))
    selected_indices = rng.sample(range(len(rows)), sample_size)
    if not args.shuffle_output:
        selected_indices = sorted(selected_indices)
    selected_rows = [rows[index] for index in selected_indices]

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_manifest, selected_rows)
    summary = {
        "train_manifest": str(train_manifest),
        "output_manifest": str(output_manifest),
        "seed": int(args.seed),
        "requested_sample_size": int(args.sample_size),
        "source_rows": len(rows),
        "sampled_rows": len(selected_rows),
        "shuffle_output": bool(args.shuffle_output),
        "selected_index_min": min(selected_indices) if selected_indices else None,
        "selected_index_max": max(selected_indices) if selected_indices else None,
    }
    if args.summary_json:
        write_json(resolve_cross_platform_path(args.summary_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
