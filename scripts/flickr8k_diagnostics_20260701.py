#!/usr/bin/env python3
"""Run July-01 Flickr8k evaluation diagnostics on lab-250.

The script is intentionally operational, not part of the training code path.
It runs a serial GPU queue for:

1. A4 normal eval vs A4 CLIP-rerank decode with lambda=0.0.
2. True/shuffled/disabled/zero-prefix image ablations for A2/A4/A5.

Outputs are written to a new diagnostics directory and do not overwrite the
previous experiment metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence


REPO_ROOT = Path("/disk5/cvlab/from252/custom-whisper-dev")
ENV_PREFIX = Path("/disk5/cvlab/from252/envs/custom-whisper-mm")
PYTHON = str(ENV_PREFIX / "bin" / "python")

TEST_MANIFEST = Path("data/flickr8k/prepared/splits/by_image_id_seed42_val10_test10/test_manifest.jsonl")
CLIP_MODEL = Path("/disk5/cvlab/from252/custom-whisper/data/models/clip/clip-vit-base-patch32")

SUITE = Path("outputs/flickr8k_decoder_prompt_large_v3")
DIAG_ROOT = Path("outputs/flickr8k_eval_diagnostics_20260701")
LOG_DIR = Path("logs/diagnostics_20260701")
STATE_PATH = LOG_DIR / "state.json"

A2_EPOCH10 = SUITE / "A2_clipseq_decoder_prompt_k16/checkpoints/epoch_10.pt"
A4_BEST = SUITE / "A4_clipseq_decoder_prompt_k16_shuffle_rank/checkpoints/best_val_loss.pt"
A5_BEST = SUITE / "A5_blip2_qformer_decoder_prompt/checkpoints/best_val_loss.pt"

NON_EVAL_CHARS_RE = re.compile(r"[^a-z0-9'\s]+")
MULTISPACE_RE = re.compile(r"\s+")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def abs_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def normalize_eval_text(text: str) -> str:
    text = str(text or "").lower().strip()
    text = NON_EVAL_CHARS_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def edit_distance(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)
    prev = list(range(len(seq_b) + 1))
    for i, token_a in enumerate(seq_a, start=1):
        cur = [i]
        for j, token_b in enumerate(seq_b, start=1):
            cost = 0 if token_a == token_b else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def metrics_done(output_root: Path) -> bool:
    path = abs_path(output_root / "metrics.json")
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if data.get("rows", data.get("count")) == 4000:
        return True
    before = data.get("before")
    return isinstance(before, dict) and before.get("count") == 4000


def eval_visspeech_cmd(checkpoint: Path, output_root: Path, extra: Optional[List[str]] = None) -> List[str]:
    cmd = [
        PYTHON,
        "scripts/eval_visspeech_custom_whisper_fuser.py",
        "--checkpoint-path",
        str(checkpoint),
        "--manifest-path",
        str(TEST_MANIFEST),
        "--output-root",
        str(output_root),
        "--batch-size",
        "8",
        "--device",
        "cuda",
        "--beam-size",
        "5",
        "--no-download",
        "--resume-from-predictions",
        "--skip-if-exists",
        "--log-every",
        "20",
    ]
    if extra:
        cmd.extend(extra)
    return cmd


def rerank_cmd(checkpoint: Path, output_root: Path) -> List[str]:
    return [
        PYTHON,
        "scripts/eval_clip_rerank.py",
        "--enable-clip-rerank",
        "--checkpoint-path",
        str(checkpoint),
        "--manifest-path",
        str(TEST_MANIFEST),
        "--output-root",
        str(output_root),
        "--clip-model-name",
        str(CLIP_MODEL),
        "--visual-model-name",
        str(CLIP_MODEL),
        "--beam-size",
        "5",
        "--rerank-n-best",
        "5",
        "--clip-rerank-lambda",
        "0.0",
        "--device",
        "cuda",
        "--no-download",
    ]


def prediction_id(row: Dict) -> str:
    return "\t".join(
        [
            str(row.get("key", "")),
            str(row.get("wav_path", "")),
            str(row.get("image_path", "")),
        ]
    )


def summarize_prediction_rows(rows: Sequence[Dict]) -> Dict[str, float]:
    total_words = total_word_edits = 0
    total_chars = total_char_edits = 0
    for row in rows:
        ref = normalize_eval_text(str(row.get("ref_text", "")))
        hyp = normalize_eval_text(str(row.get("pred_text", "")))
        ref_words = ref.split()
        hyp_words = hyp.split()
        total_words += max(1, len(ref_words))
        total_word_edits += edit_distance(ref_words, hyp_words)
        total_chars += max(1, len(ref))
        total_char_edits += edit_distance(list(ref), list(hyp))
    return {
        "count": len(rows),
        "wer": total_word_edits / total_words if total_words else 0.0,
        "cer": total_char_edits / total_chars if total_chars else 0.0,
        "word_edits": total_word_edits,
        "words": total_words,
        "char_edits": total_char_edits,
        "chars": total_chars,
    }


def diff_a4_outputs() -> bool:
    normal_path = abs_path(DIAG_ROOT / "a4_same_ckpt_normal_beam5/predictions.jsonl")
    rerank_path = abs_path(DIAG_ROOT / "a4_same_ckpt_rerank_lambda0_beam5/predictions_before.jsonl")
    out_dir = abs_path(DIAG_ROOT / "a4_normal_vs_rerank_lambda0_diff")
    if not normal_path.is_file() or not rerank_path.is_file():
        return False

    normal_rows = read_jsonl(normal_path)
    rerank_rows = read_jsonl(rerank_path)
    normal_by_id = {prediction_id(row): row for row in normal_rows}
    rerank_by_id = {prediction_id(row): row for row in rerank_rows}
    normal_ids = [prediction_id(row) for row in normal_rows]
    rerank_ids = [prediction_id(row) for row in rerank_rows]

    common_ids = [row_id for row_id in normal_ids if row_id in rerank_by_id]
    disagreements: List[Dict] = []
    ref_mismatches: List[Dict] = []
    for row_id in common_ids:
        a = normal_by_id[row_id]
        b = rerank_by_id[row_id]
        ref_a = normalize_eval_text(str(a.get("ref_text", "")))
        ref_b = normalize_eval_text(str(b.get("ref_text", "")))
        hyp_a = normalize_eval_text(str(a.get("pred_text", "")))
        hyp_b = normalize_eval_text(str(b.get("pred_text", "")))
        if ref_a != ref_b:
            ref_mismatches.append(
                {
                    "key": a.get("key"),
                    "normal_ref": a.get("ref_text"),
                    "rerank_ref": b.get("ref_text"),
                    "normal_norm_ref": ref_a,
                    "rerank_norm_ref": ref_b,
                }
            )
        if hyp_a != hyp_b:
            disagreements.append(
                {
                    "key": a.get("key"),
                    "wav_path": a.get("wav_path"),
                    "image_path": a.get("image_path"),
                    "ref_text": a.get("ref_text"),
                    "normal_pred_text": a.get("pred_text"),
                    "rerank_before_pred_text": b.get("pred_text"),
                    "normal_norm_pred": hyp_a,
                    "rerank_before_norm_pred": hyp_b,
                    "normal_word_edits": edit_distance(ref_a.split(), hyp_a.split()),
                    "rerank_word_edits": edit_distance(ref_b.split(), hyp_b.split()),
                }
            )

    summary = {
        "normal_predictions": str(normal_path),
        "rerank_before_predictions": str(rerank_path),
        "normal_count": len(normal_rows),
        "rerank_count": len(rerank_rows),
        "normal_unique_ids": len(normal_by_id),
        "rerank_unique_ids": len(rerank_by_id),
        "same_order_ids": normal_ids == rerank_ids,
        "missing_in_rerank": len([row_id for row_id in normal_ids if row_id not in rerank_by_id]),
        "missing_in_normal": len([row_id for row_id in rerank_ids if row_id not in normal_by_id]),
        "ref_mismatch_count": len(ref_mismatches),
        "hyp_mismatch_count": len(disagreements),
        "normal_metrics_recomputed": summarize_prediction_rows(normal_rows),
        "rerank_before_metrics_recomputed": summarize_prediction_rows(rerank_rows),
    }
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "hyp_disagreements.jsonl", disagreements)
    write_jsonl(out_dir / "ref_mismatches.jsonl", ref_mismatches)
    return True


@dataclass
class Task:
    name: str
    command: Optional[List[str]]
    done: Callable[[], bool]
    is_cpu: bool = False


def build_tasks() -> List[Task]:
    image_tasks = []
    specs = [
        ("A2_epoch10", A2_EPOCH10),
        ("A4_10_best", A4_BEST),
        ("A5_20_best", A5_BEST),
    ]
    modes = [
        ("true", []),
        ("shuffle", ["--shuffle-images-at-eval"]),
        ("disable", ["--disable-image-at-eval"]),
        ("zero_prefix", ["--zero-prefix-at-eval"]),
    ]
    for exp_name, ckpt in specs:
        for mode_name, extra in modes:
            out = DIAG_ROOT / "image_ablation" / exp_name / mode_name
            image_tasks.append(
                Task(
                    f"ablation_{exp_name}_{mode_name}",
                    eval_visspeech_cmd(ckpt, out, extra),
                    lambda out=out: metrics_done(out),
                )
            )

    return [
        Task(
            "a4_same_ckpt_normal_beam5",
            eval_visspeech_cmd(A4_BEST, DIAG_ROOT / "a4_same_ckpt_normal_beam5"),
            lambda: metrics_done(DIAG_ROOT / "a4_same_ckpt_normal_beam5"),
        ),
        Task(
            "a4_same_ckpt_rerank_lambda0_beam5",
            rerank_cmd(A4_BEST, DIAG_ROOT / "a4_same_ckpt_rerank_lambda0_beam5"),
            lambda: metrics_done(DIAG_ROOT / "a4_same_ckpt_rerank_lambda0_beam5"),
        ),
        Task("diff_a4_normal_vs_rerank_lambda0", None, diff_a4_outputs, is_cpu=True),
        *image_tasks,
    ]


class Runner:
    def __init__(self, gpu: int, tasks: Sequence[Task]):
        self.gpu = gpu
        self.tasks = list(tasks)
        self.state_path = abs_path(STATE_PATH)
        self.log_dir = abs_path(LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state = self.load_state()

    def load_state(self) -> Dict[str, Dict]:
        if self.state_path.is_file():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{now()}] {message}"
        print(line, flush=True)
        with (self.log_dir / "runner.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def set_status(self, task: str, status: str, **extra) -> None:
        row = self.state.setdefault(task, {})
        row.update(extra)
        row["status"] = status
        row["updated_at"] = now()
        self.save_state()

    def run_shell_task(self, task: Task) -> int:
        assert task.command is not None
        log_path = self.log_dir / f"{task.name}.log"
        env = os.environ.copy()
        env["PATH"] = f"{ENV_PREFIX / 'bin'}:{env.get('PATH', '')}"
        env["CONDA_PREFIX"] = str(ENV_PREFIX)
        env["PYTHONNOUSERSITE"] = "1"
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        with log_path.open("a", buffering=1, encoding="utf-8") as log:
            log.write(f"\n\n[{now()}] diagnostics starting {task.name} on GPU {self.gpu}\n")
            log.write("COMMAND: " + " ".join(task.command) + "\n")
            process = subprocess.Popen(
                task.command,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.set_status(task.name, "running", pid=process.pid, gpu=self.gpu, log=str(log_path))
            self.log(f"{task.name}: started pid={process.pid} gpu={self.gpu}")
            code = process.wait()
            log.write(f"[{now()}] diagnostics observed exit code {code}\n")
        return code

    def run(self) -> int:
        self.log(f"diagnostics started; gpu={self.gpu}")
        for task in self.tasks:
            if task.done():
                self.set_status(task.name, "done", reason="already-complete")
                self.log(f"{task.name}: already done")
                continue
            if task.is_cpu:
                self.set_status(task.name, "running")
                ok = task.done()
                self.set_status(task.name, "done" if ok else "failed")
                self.log(f"{task.name}: {'done' if ok else 'failed'}")
                if not ok:
                    return 1
                continue
            code = self.run_shell_task(task)
            if code != 0:
                self.set_status(task.name, "failed", exit_code=code)
                self.log(f"{task.name}: failed exit_code={code}")
                return code
            if not task.done():
                self.set_status(task.name, "failed", exit_code=code, reason="completion-condition-false")
                self.log(f"{task.name}: exited 0 but completion condition is false")
                return 1
            self.set_status(task.name, "done", exit_code=code)
            self.log(f"{task.name}: done")
            time.sleep(5)
        self.log("diagnostics finished")
        return 0


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    tasks = build_tasks()
    if args.dry_run:
        for task in tasks:
            print(f"- {task.name} done={task.done()} cpu={task.is_cpu}")
            if task.command:
                print("  " + " ".join(task.command))
        return 0
    return Runner(args.gpu, tasks).run()


if __name__ == "__main__":
    raise SystemExit(main())
