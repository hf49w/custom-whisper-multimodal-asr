from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from visspeech_custom_whisper_utils import (
    build_ordered_fieldnames,
    ensure_dir,
    load_manifest,
    resolve_cross_platform_path,
    write_csv_rows,
    write_jsonl,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_FLICKR8K_PREPARED_ROOT = REPO_ROOT / "data" / "flickr8k" / "prepared"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a prepared manifest into train/val/test partitions. "
            "Rows are grouped by a field such as image_id so related rows stay together."
        )
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=str(DEFAULT_FLICKR8K_PREPARED_ROOT / "manifest.jsonl"),
        help="Prepared manifest JSONL/CSV.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Output directory for split manifests.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of rows assigned to validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of rows assigned to test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible group order.",
    )
    parser.add_argument(
        "--group-by-field",
        type=str,
        default="image_id",
        help="Field used to keep related rows in the same split. Use empty string to split by row.",
    )
    return parser.parse_args()


def assign_groups_to_target_rows(
    group_items: Sequence[Tuple[str, List[Dict[str, Any]]]],
    *,
    target_rows: int,
) -> Tuple[List[Dict[str, Any]], List[str], List[Tuple[str, List[Dict[str, Any]]]]]:
    assigned_rows: List[Dict[str, Any]] = []
    assigned_keys: List[str] = []
    remaining_items: List[Tuple[str, List[Dict[str, Any]]]] = []
    running_rows = 0
    total_groups = len(group_items)

    for index, (group_key, group_rows) in enumerate(group_items):
        remaining_groups_after = total_groups - index - 1
        should_assign = running_rows < target_rows and remaining_groups_after > 0
        if should_assign:
            assigned_rows.extend(group_rows)
            assigned_keys.append(group_key)
            running_rows += len(group_rows)
        else:
            remaining_items.append((group_key, group_rows))

    return assigned_rows, assigned_keys, remaining_items


def main() -> None:
    args = parse_args()
    if args.val_ratio < 0 or args.test_ratio < 0:
        raise ValueError("val_ratio and test_ratio must be >= 0")
    if args.val_ratio + args.test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be < 1")

    manifest_path = resolve_cross_platform_path(args.manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = load_manifest(manifest_path)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    group_by_field = str(args.group_by_field or "").strip()
    output_root = (
        resolve_cross_platform_path(args.output_root)
        if args.output_root
        else manifest_path.parent
        / (
            f"splits/by_{group_by_field or 'row'}_seed{args.seed}"
            f"_val{int(args.val_ratio * 100):02d}_test{int(args.test_ratio * 100):02d}"
        )
    )
    ensure_dir(output_root)

    groups: Dict[str, List[Dict[str, Any]]] = {}
    if group_by_field:
        for row in rows:
            group_key = str(row.get(group_by_field) or row.get("key") or "")
            groups.setdefault(group_key, []).append(row)
    else:
        for index, row in enumerate(rows):
            groups[f"row-{index:05d}"] = [row]

    group_items = list(groups.items())
    rng = random.Random(args.seed)
    rng.shuffle(group_items)

    total_rows = len(rows)
    target_val_rows = max(1, round(total_rows * args.val_ratio)) if args.val_ratio > 0 else 0
    target_test_rows = max(1, round(total_rows * args.test_ratio)) if args.test_ratio > 0 else 0

    test_rows, test_group_keys, remaining_after_test = assign_groups_to_target_rows(
        group_items,
        target_rows=target_test_rows,
    )
    val_rows, val_group_keys, remaining_after_val = assign_groups_to_target_rows(
        remaining_after_test,
        target_rows=target_val_rows,
    )
    train_rows = [row for _, group_rows in remaining_after_val for row in group_rows]

    if not train_rows:
        raise ValueError("Train split is empty; adjust val/test ratios.")
    if args.val_ratio > 0 and not val_rows:
        raise ValueError("Validation split is empty; adjust val_ratio.")
    if args.test_ratio > 0 and not test_rows:
        raise ValueError("Test split is empty; adjust test_ratio.")

    fieldnames = build_ordered_fieldnames(rows)
    train_manifest_path = output_root / "train_manifest.jsonl"
    val_manifest_path = output_root / "val_manifest.jsonl"
    test_manifest_path = output_root / "test_manifest.jsonl"
    train_csv_path = output_root / "train_manifest.csv"
    val_csv_path = output_root / "val_manifest.csv"
    test_csv_path = output_root / "test_manifest.csv"
    stats_path = output_root / "split_stats.json"

    write_jsonl(train_manifest_path, train_rows)
    write_jsonl(val_manifest_path, val_rows)
    write_jsonl(test_manifest_path, test_rows)
    write_csv_rows(train_csv_path, train_rows, fieldnames)
    write_csv_rows(val_csv_path, val_rows, fieldnames)
    write_csv_rows(test_csv_path, test_rows, fieldnames)

    stats = {
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "seed": args.seed,
        "group_by_field": group_by_field,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "total_rows": total_rows,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "total_groups": len(groups),
        "val_groups": len(val_group_keys),
        "test_groups": len(test_group_keys),
        "train_manifest_path": str(train_manifest_path),
        "val_manifest_path": str(val_manifest_path),
        "test_manifest_path": str(test_manifest_path),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[DONE] train_rows={len(train_rows)} val_rows={len(val_rows)} test_rows={len(test_rows)}"
    )
    print(f"[DONE] train_manifest={train_manifest_path}")
    print(f"[DONE] val_manifest={val_manifest_path}")
    print(f"[DONE] test_manifest={test_manifest_path}")
    print(f"[DONE] stats={stats_path}")


if __name__ == "__main__":
    main()
