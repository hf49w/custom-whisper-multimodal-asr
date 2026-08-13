#!/usr/bin/env python3
"""Run the 2026-06-30 Flickr8k follow-up experiment queue on lab-250.

This is a small operational queue for the training server.  It intentionally
does not delete or mirror any runtime output.  It waits for one of the allowed
GPUs to become free, runs one command per free GPU, and restarts interrupted
commands from the existing resumable artifacts where possible.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional


REPO_ROOT = Path("/disk5/cvlab/from252/custom-whisper-dev")
ENV_PREFIX = Path("/disk5/cvlab/from252/envs/custom-whisper-mm")
PYTHON = str(ENV_PREFIX / "bin" / "python")

SUITE = Path("outputs/flickr8k_decoder_prompt_large_v3")
A1_SUITE = Path("outputs/flickr8k_decoder_prompt_large_v3_A1_rerun_20260626_165442")
EVAL_SUITE = Path("outputs/flickr8k_decoder_prompt_large_v3_eval")

TRAIN_MANIFEST = Path("data/flickr8k/prepared/splits/by_image_id_seed42_val10_test10/train_manifest.jsonl")
VAL_MANIFEST = Path("data/flickr8k/prepared/splits/by_image_id_seed42_val10_test10/val_manifest.jsonl")
TEST_MANIFEST = Path("data/flickr8k/prepared/splits/by_image_id_seed42_val10_test10/test_manifest.jsonl")
WHISPER_MODEL = Path("/disk5/cvlab/from252/custom-whisper/data/models/whisper/large-v3.pt")
CLIP_MODEL = Path("/disk5/cvlab/from252/custom-whisper/data/models/clip/clip-vit-base-patch32")

LOG_DIR = Path("logs/queue_20260630")
STATE_PATH = Path("logs/queue_20260630_state.json")

GPU_FREE_USED_MIB = 1024


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def rel(path: Path) -> str:
    return str(path)


def abs_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def read_json(path: Path):
    with abs_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def max_epoch(output_root: Path) -> int:
    history_path = output_root / "train_history.json"
    try:
        history = read_json(history_path)
    except Exception:
        return 0
    epochs: List[int] = []
    if isinstance(history, list):
        for item in history:
            if isinstance(item, dict) and isinstance(item.get("epoch"), int):
                epochs.append(item["epoch"])
    return max(epochs) if epochs else 0


def train_done(output_root: Path, target_epoch: int) -> bool:
    ckpt = output_root / "checkpoints" / "best_val_loss.pt"
    return max_epoch(output_root) >= target_epoch and abs_path(ckpt).is_file()


def metrics_done(output_root: Path) -> bool:
    path = output_root / "metrics.json"
    full = abs_path(path)
    if not full.is_file() or full.stat().st_size <= 0:
        return False
    try:
        data = read_json(path)
    except Exception:
        return False
    rows = data.get("rows", data.get("count"))
    if rows == 4000:
        return True
    # CLIP rerank metrics are nested by lambda; accept the canonical 4000-count
    # structure as complete.
    before = data.get("before")
    if isinstance(before, dict) and before.get("count") == 4000:
        return True
    return False


def checkpoint_exists(output_root: Path) -> bool:
    return abs_path(output_root / "checkpoints" / "last.pt").is_file()


def eval_visspeech_cmd(checkpoint: Path, output_root: Path) -> List[str]:
    return [
        PYTHON,
        "scripts/eval_visspeech_custom_whisper_fuser.py",
        "--checkpoint-path",
        rel(checkpoint),
        "--manifest-path",
        rel(TEST_MANIFEST),
        "--output-root",
        rel(output_root),
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


def train_base_cmd(output_root: Path, target_epoch: int, lr: str) -> List[str]:
    return [
        PYTHON,
        "scripts/train_visspeech_custom_whisper_fuser.py",
        "--train-manifest",
        rel(TRAIN_MANIFEST),
        "--val-manifest",
        rel(VAL_MANIFEST),
        "--whisper-model",
        rel(WHISPER_MODEL),
        "--epochs",
        str(target_epoch),
        "--batch-size",
        "8",
        "--device",
        "cuda",
        "--freeze-whisper",
        "--freeze-visual-encoder",
        "--no-download",
        "--output-root",
        rel(output_root),
        "--lr",
        lr,
        "--resume-from",
        rel(output_root / "checkpoints" / "last.pt"),
    ]


def train_a1_cmd(output_root: Path, target_epoch: int) -> List[str]:
    cmd = train_base_cmd(output_root, target_epoch, "1e-3")
    insert = [
        "--visual-encoder",
        "none",
        "--visual-fuser",
        "select_speech",
        "--fusion-location",
        "decoder_prefix",
        "--decoder-prompt-adapter",
        "blank_prefix",
        "--decoder-prompt-len",
        "16",
    ]
    return cmd[:-4] + insert + cmd[-4:]


def train_clip_prompt_cmd(output_root: Path, target_epoch: int, prompt_len: int) -> List[str]:
    cmd = train_base_cmd(output_root, target_epoch, "1e-3")
    insert = [
        "--visual-encoder",
        "clip",
        "--clip-model-name",
        rel(CLIP_MODEL),
        "--clip-return-sequence",
        "--visual-fuser",
        "select_speech",
        "--fusion-location",
        "decoder_prefix",
        "--decoder-prompt-adapter",
        "resampler",
        "--decoder-prompt-len",
        str(prompt_len),
    ]
    return cmd[:-4] + insert + cmd[-4:]


def a0_baseline_cmd(output_root: Path) -> List[str]:
    return [
        PYTHON,
        "scripts/eval_whisper_baseline.py",
        "--whisper-model",
        rel(WHISPER_MODEL),
        "--manifest-path",
        rel(TEST_MANIFEST),
        "--output-root",
        rel(output_root),
        "--device",
        "cuda",
        "--beam-size",
        "5",
    ]


def a7_rerank_cmd(checkpoint: Path, output_root: Path) -> List[str]:
    return [
        PYTHON,
        "scripts/eval_clip_rerank.py",
        "--enable-clip-rerank",
        "--checkpoint-path",
        rel(checkpoint),
        "--manifest-path",
        rel(TEST_MANIFEST),
        "--output-root",
        rel(output_root),
        "--clip-model-name",
        rel(CLIP_MODEL),
        "--visual-model-name",
        rel(CLIP_MODEL),
        "--beam-size",
        "10",
        "--rerank-n-best",
        "5",
        "--clip-rerank-lambda",
        "0.05",
        "0.1",
        "0.2",
        "--device",
        "cuda",
        "--no-download",
    ]


@dataclass
class Task:
    name: str
    command_factory: Optional[Callable[[], List[str]]]
    done: Callable[[], bool]
    deps: List[str] = field(default_factory=list)
    restartable: bool = True
    max_restarts: int = 12
    min_restart_seconds: int = 600


@dataclass
class Running:
    task: Task
    process: subprocess.Popen
    gpu: int
    log_file: object
    started_at: float


class QueueRunner:
    def __init__(self, tasks: List[Task], allowed_gpus: List[int], interval: int):
        self.tasks = tasks
        self.allowed_gpus = allowed_gpus
        self.interval = interval
        self.running: Dict[str, Running] = {}
        self.state = self.load_state()
        self.queue_log_path = abs_path(LOG_DIR / "queue.log")
        abs_path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"[{now()}] {message}"
        print(line, flush=True)
        with self.queue_log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def load_state(self) -> Dict[str, dict]:
        full = abs_path(STATE_PATH)
        if full.exists():
            try:
                return json.loads(full.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def save_state(self) -> None:
        full = abs_path(STATE_PATH)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(json.dumps(self.state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def task_state(self, name: str) -> dict:
        return self.state.setdefault(name, {"status": "pending", "restarts": 0})

    def set_status(self, name: str, status: str, **extra) -> None:
        st = self.task_state(name)
        st["status"] = status
        st.update(extra)
        st["updated_at"] = now()
        self.save_state()

    def completed(self, task: Task) -> bool:
        if task.done():
            if self.task_state(task.name).get("status") != "done":
                self.set_status(task.name, "done", reason="completion-condition")
                self.log(f"{task.name}: done")
            return True
        return self.task_state(task.name).get("status") == "done"

    def deps_done(self, task: Task) -> bool:
        return all(self.task_state(dep).get("status") == "done" for dep in task.deps)

    def busy_gpus(self) -> set:
        return {r.gpu for r in self.running.values()}

    def free_gpus(self) -> List[int]:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            cwd=str(REPO_ROOT),
            text=True,
        )
        used: Dict[int, int] = {}
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    used[int(parts[0])] = int(parts[1])
                except ValueError:
                    pass
        busy = self.busy_gpus()
        return [
            gpu
            for gpu in self.allowed_gpus
            if gpu not in busy and used.get(gpu, 10**9) <= GPU_FREE_USED_MIB
        ]

    def start_task(self, task: Task, gpu: int) -> None:
        assert task.command_factory is not None
        command = task.command_factory()
        log_path = abs_path(LOG_DIR / f"{task.name}.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", buffering=1, encoding="utf-8")
        log_file.write(f"\n\n[{now()}] queue starting {task.name} on GPU {gpu}\n")
        log_file.write("COMMAND: " + " ".join(command) + "\n")
        env = os.environ.copy()
        env["PATH"] = f"{ENV_PREFIX / 'bin'}:{env.get('PATH', '')}"
        env["CONDA_PREFIX"] = str(ENV_PREFIX)
        env["PYTHONNOUSERSITE"] = "1"
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.running[task.name] = Running(task, process, gpu, log_file, time.time())
        self.set_status(task.name, "running", pid=process.pid, gpu=gpu, log=str(log_path))
        self.log(f"{task.name}: started pid={process.pid} gpu={gpu} log={log_path}")

    def poll_running(self) -> None:
        for name, run in list(self.running.items()):
            code = run.process.poll()
            if code is None:
                continue
            run.log_file.write(f"[{now()}] queue observed exit code {code}\n")
            run.log_file.close()
            del self.running[name]
            if code == 0 and run.task.done():
                self.set_status(name, "done", exit_code=code)
                self.log(f"{name}: completed")
                continue
            if code == 0:
                self.set_status(name, "failed", exit_code=code, reason="exit-zero-but-completion-condition-false")
                self.log(f"{name}: exited 0 but completion condition is false")
                continue
            st = self.task_state(name)
            restarts = int(st.get("restarts", 0))
            if not run.task.restartable or restarts >= run.task.max_restarts:
                self.set_status(name, "failed", exit_code=code, restarts=restarts)
                self.log(f"{name}: failed exit_code={code}; no more restarts")
                continue
            self.set_status(
                name,
                "pending",
                exit_code=code,
                restarts=restarts + 1,
                not_before=time.time() + run.task.min_restart_seconds,
            )
            self.log(f"{name}: exited {code}; restart {restarts + 1}/{run.task.max_restarts} after cooldown")

    def maybe_start_ready(self) -> None:
        free = self.free_gpus()
        if not free:
            return
        for task in self.tasks:
            if not free:
                return
            if task.name in self.running:
                continue
            st = self.task_state(task.name)
            if st.get("status") in {"running", "done", "failed"}:
                continue
            if self.completed(task):
                continue
            if not self.deps_done(task):
                continue
            if task.command_factory is None:
                # Barrier task whose completion condition is not true yet.
                continue
            not_before = float(st.get("not_before", 0) or 0)
            if time.time() < not_before:
                continue
            gpu = free.pop(0)
            self.start_task(task, gpu)

    def all_done_or_failed(self) -> bool:
        return all(self.task_state(t.name).get("status") in {"done", "failed"} for t in self.tasks)

    def run(self) -> int:
        self.log(f"queue started; allowed_gpus={self.allowed_gpus}")
        for task in self.tasks:
            self.task_state(task.name)
        self.save_state()
        while True:
            for task in self.tasks:
                if task.name not in self.running:
                    self.completed(task)
            self.poll_running()
            self.maybe_start_ready()
            if self.all_done_or_failed() and not self.running:
                self.log("queue finished")
                return 0
            time.sleep(self.interval)


def build_tasks() -> List[Task]:
    a1_root = A1_SUITE / "A1_blank_decoder_prefix_k16"
    a2_root = SUITE / "A2_clipseq_decoder_prompt_k16"
    a3_root = SUITE / "A3_clipseq_decoder_prompt_k32"
    a4_root = SUITE / "A4_clipseq_decoder_prompt_k16_shuffle_rank"
    a5_root = SUITE / "A5_blip2_qformer_decoder_prompt"

    return [
        Task(
            "eval_A0_baseline",
            lambda: a0_baseline_cmd(EVAL_SUITE / "A0_whisper_baseline"),
            lambda: metrics_done(EVAL_SUITE / "A0_whisper_baseline"),
            restartable=True,
            max_restarts=3,
        ),
        Task("wait_A3_10", None, lambda: train_done(a3_root, 10)),
        Task("wait_A4_10", None, lambda: train_done(a4_root, 10)),
        Task("wait_A5_20", None, lambda: train_done(a5_root, 20)),
        Task(
            "eval_A3_10",
            lambda: eval_visspeech_cmd(a3_root / "checkpoints" / "best_val_loss.pt", EVAL_SUITE / "A3_10ep_best_val"),
            lambda: metrics_done(EVAL_SUITE / "A3_10ep_best_val"),
            deps=["wait_A3_10"],
        ),
        Task(
            "eval_A4_10",
            lambda: eval_visspeech_cmd(a4_root / "checkpoints" / "best_val_loss.pt", EVAL_SUITE / "A4_10ep_best_val"),
            lambda: metrics_done(EVAL_SUITE / "A4_10ep_best_val"),
            deps=["wait_A4_10"],
        ),
        Task(
            "eval_A7_from_A4_10",
            lambda: a7_rerank_cmd(a4_root / "checkpoints" / "best_val_loss.pt", EVAL_SUITE / "A7_from_A4_10_clip_rerank"),
            lambda: metrics_done(EVAL_SUITE / "A7_from_A4_10_clip_rerank"),
            deps=["wait_A4_10"],
            restartable=True,
            max_restarts=3,
        ),
        Task(
            "eval_A5_20",
            lambda: eval_visspeech_cmd(a5_root / "checkpoints" / "best_val_loss.pt", EVAL_SUITE / "A5_20ep_best_val"),
            lambda: metrics_done(EVAL_SUITE / "A5_20ep_best_val"),
            deps=["wait_A5_20"],
        ),
        Task(
            "train_A1_20",
            lambda: train_a1_cmd(a1_root, 20),
            lambda: train_done(a1_root, 20),
            restartable=True,
        ),
        Task(
            "eval_A1_20",
            lambda: eval_visspeech_cmd(a1_root / "checkpoints" / "best_val_loss.pt", EVAL_SUITE / "A1_20ep_best_val"),
            lambda: metrics_done(EVAL_SUITE / "A1_20ep_best_val"),
            deps=["train_A1_20"],
        ),
        Task(
            "train_A2_20",
            lambda: train_clip_prompt_cmd(a2_root, 20, 16),
            lambda: train_done(a2_root, 20),
            restartable=True,
        ),
        Task(
            "eval_A2_20",
            lambda: eval_visspeech_cmd(a2_root / "checkpoints" / "best_val_loss.pt", EVAL_SUITE / "A2_20ep_best_val"),
            lambda: metrics_done(EVAL_SUITE / "A2_20ep_best_val"),
            deps=["train_A2_20"],
        ),
        Task(
            "train_A3_20",
            lambda: train_clip_prompt_cmd(a3_root, 20, 32),
            lambda: train_done(a3_root, 20),
            deps=["eval_A3_10"],
            restartable=True,
        ),
        Task(
            "eval_A3_20",
            lambda: eval_visspeech_cmd(a3_root / "checkpoints" / "best_val_loss.pt", EVAL_SUITE / "A3_20ep_best_val"),
            lambda: metrics_done(EVAL_SUITE / "A3_20ep_best_val"),
            deps=["train_A3_20"],
        ),
    ]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowed-gpus", default="5,0,1,3")
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    allowed_gpus = [int(x) for x in args.allowed_gpus.split(",") if x.strip()]
    tasks = build_tasks()
    if args.dry_run:
        print("tasks:")
        for task in tasks:
            print(f"- {task.name} deps={task.deps} done={task.done()}")
            if task.command_factory is not None:
                print("  " + " ".join(task.command_factory()))
        return 0
    runner = QueueRunner(tasks, allowed_gpus, args.interval)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
