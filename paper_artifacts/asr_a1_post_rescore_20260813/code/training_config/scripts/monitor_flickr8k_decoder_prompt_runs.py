#!/usr/bin/env python3
"""Watch Flickr8k decoder-prompt training jobs and resume them if they stop.

This is intended to run on the training server, not on the source-only local
mirror.  It does not modify model code or delete outputs.  For each configured
experiment it:

1. checks whether a matching training process is already alive;
2. checks whether the experiment has already reached the target epoch count;
3. if it is not alive and not done, starts the same training command with
   ``--resume-from <output_root>/checkpoints/last.pt``.

The monitor is deliberately conservative: by default it requires ``last.pt`` to
exist before restarting a job, so an accidentally wrong output path does not
start a new run from scratch.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_ENV_PREFIX = "/disk5/cvlab/from252/envs/custom-whisper-mm"
DEFAULT_REPO_ROOT = "/disk5/cvlab/from252/custom-whisper-dev"
DEFAULT_SUITE_ROOT = "outputs/flickr8k_decoder_prompt_large_v3"
DEFAULT_DATA_SPLIT = "data/flickr8k/prepared/splits/by_image_id_seed42_val10_test10"
DEFAULT_WHISPER_MODEL = "/disk5/cvlab/from252/custom-whisper/data/models/whisper/large-v3.pt"
DEFAULT_CLIP_MODEL = "/disk5/cvlab/from252/custom-whisper/data/models/clip/clip-vit-base-patch32"
DEFAULT_BLIP2_MODEL = "/disk5/cvlab/from252/custom-whisper/data/models/blip2/blip2-opt-2.7b"


@dataclass(frozen=True)
class JobSpec:
    name: str
    gpu: str
    output_root: Path
    args: Tuple[str, ...]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def resolve_path(path: str | Path, repo_root: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def parse_key_value_overrides(values: Sequence[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=VALUE override, got: {value!r}")
        name, raw = value.split("=", 1)
        name = name.strip().upper()
        if not name:
            raise ValueError(f"Empty job name in override: {value!r}")
        parsed[name] = raw.strip()
    return parsed


def parse_gpu_map(values: Sequence[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"Expected NAME:GPU mapping, got: {value!r}")
        name, gpu = value.split(":", 1)
        name = name.strip().upper()
        gpu = gpu.strip()
        if not name or not gpu:
            raise ValueError(f"Invalid GPU mapping: {value!r}")
        parsed[name] = gpu
    return parsed


def csv_names(raw: str) -> List[str]:
    names = [item.strip().upper() for item in raw.split(",") if item.strip()]
    if not names:
        raise ValueError("--jobs must contain at least one job name")
    return names


def command_has_option(cmdline: Sequence[str], option: str, expected_value: Path) -> bool:
    expected = str(expected_value.resolve())
    for index, token in enumerate(cmdline):
        if token == option and index + 1 < len(cmdline):
            candidate = Path(cmdline[index + 1])
            if not candidate.is_absolute():
                # Training is always launched from repo root by this monitor.
                cwd = read_proc_cwd(cmdline)
                candidate = (cwd / candidate) if cwd is not None else candidate
            try:
                if str(candidate.resolve()) == expected:
                    return True
            except OSError:
                if str(candidate) == expected:
                    return True
        prefix = option + "="
        if token.startswith(prefix):
            candidate = Path(token[len(prefix) :])
            try:
                if str(candidate.resolve()) == expected:
                    return True
            except OSError:
                if str(candidate) == expected:
                    return True
    return False


def read_proc_cwd(_cmdline: Sequence[str]) -> Optional[Path]:
    # Kept as a separate function because /proc/<pid>/cwd is only available
    # while iterating a concrete process; command_has_option may be called from
    # tests without a process id.  Relative output roots are handled by also
    # checking suffixes in find_live_training_processes.
    return None


def iter_process_cmdlines() -> Iterable[Tuple[int, List[str]]]:
    proc_root = Path("/proc")
    for child in proc_root.iterdir():
        if not child.name.isdigit():
            continue
        cmdline_path = child / "cmdline"
        try:
            raw = cmdline_path.read_bytes()
        except (OSError, PermissionError):
            continue
        if not raw:
            continue
        parts = [p.decode("utf-8", errors="replace") for p in raw.split(b"\0") if p]
        if parts:
            yield int(child.name), parts


def find_live_training_processes(output_root: Path) -> List[int]:
    output_abs = str(output_root.resolve())
    output_text = str(output_root)
    pids: List[int] = []
    for pid, cmdline in iter_process_cmdlines():
        joined = " ".join(cmdline)
        if "train_visspeech_custom_whisper_fuser.py" not in joined:
            continue
        if output_abs in joined or output_text in joined:
            pids.append(pid)
            continue
        # Fallback: parse --output-root and compare basename/path suffix.  This
        # handles commands launched with relative output paths.
        for index, token in enumerate(cmdline):
            value: Optional[str] = None
            if token == "--output-root" and index + 1 < len(cmdline):
                value = cmdline[index + 1]
            elif token.startswith("--output-root="):
                value = token.split("=", 1)[1]
            if value and (
                output_abs.endswith(value)
                or str(Path(value)).endswith(str(output_root))
                or Path(value).name == output_root.name
            ):
                pids.append(pid)
                break
    return sorted(set(pids))


def load_checkpoint_summary(checkpoint_path: Path, python_exe: Path) -> Dict[str, object]:
    if not checkpoint_path.is_file():
        return {"exists": False}
    code = r"""
import json
import sys
import torch

path = sys.argv[1]
ckpt = torch.load(path, map_location="cpu")
history = ckpt.get("train_history") or []
cfg = ckpt.get("train_config") or {}
print(json.dumps({
    "exists": True,
    "resume_epoch": ckpt.get("resume_epoch"),
    "resume_batch_index": ckpt.get("resume_batch_index"),
    "global_step": ckpt.get("global_step"),
    "history_len": len(history),
    "max_history_epoch": max([int(item.get("epoch", 0)) for item in history] or [0]),
    "config_epochs": cfg.get("epochs"),
}, ensure_ascii=False))
"""
    proc = subprocess.run(
        [str(python_exe), "-c", code, str(checkpoint_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "exists": True,
            "error": proc.stderr.strip() or proc.stdout.strip() or f"exit={proc.returncode}",
        }
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return {"exists": True, "error": "checkpoint inspector produced no output"}
    return json.loads(lines[-1])


def is_done(summary: Dict[str, object], target_epochs: int) -> bool:
    if not summary.get("exists"):
        return False
    resume_epoch = summary.get("resume_epoch")
    resume_batch_index = summary.get("resume_batch_index")
    max_history_epoch = int(summary.get("max_history_epoch") or 0)
    if max_history_epoch >= target_epochs:
        return True
    if isinstance(resume_epoch, int) and resume_epoch > target_epochs and int(resume_batch_index or 0) == 0:
        return True
    return False


def build_jobs(
    *,
    repo_root: Path,
    suite_root: Path,
    jobs: Sequence[str],
    gpu_map: Dict[str, str],
    output_overrides: Dict[str, str],
    train_manifest: str,
    val_manifest: str,
    whisper_model: str,
    clip_model: str,
    blip2_model: str,
    target_epochs: int,
    batch_size: int,
) -> List[JobSpec]:
    def out(name: str, default_rel: str) -> Path:
        override = output_overrides.get(name)
        if override:
            return resolve_path(override, repo_root)
        return resolve_path(suite_root / default_rel, repo_root)

    base = (
        "scripts/train_visspeech_custom_whisper_fuser.py",
        "--train-manifest",
        train_manifest,
        "--val-manifest",
        val_manifest,
        "--whisper-model",
        whisper_model,
        "--epochs",
        str(target_epochs),
        "--batch-size",
        str(batch_size),
        "--device",
        "cuda",
        "--freeze-whisper",
        "--freeze-visual-encoder",
        "--no-download",
    )

    specs: Dict[str, Tuple[Path, Tuple[str, ...]]] = {}

    a1_out = out("A1", "A1_blank_decoder_prefix_k16")
    specs["A1"] = (
        a1_out,
        base
        + (
            "--output-root",
            str(a1_out),
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
            "--lr",
            "1e-3",
        ),
    )

    for name, prompt_len in (("A2", "16"), ("A3", "32")):
        exp_out = out(name, f"{name}_clipseq_decoder_prompt_k{prompt_len}")
        specs[name] = (
            exp_out,
            base
            + (
                "--output-root",
                str(exp_out),
                "--visual-encoder",
                "clip",
                "--clip-model-name",
                clip_model,
                "--clip-return-sequence",
                "--visual-fuser",
                "select_speech",
                "--fusion-location",
                "decoder_prefix",
                "--decoder-prompt-adapter",
                "resampler",
                "--decoder-prompt-len",
                prompt_len,
                "--lr",
                "1e-3",
            ),
        )

    a4_out = out("A4", "A4_clipseq_decoder_prompt_k16_shuffle_rank")
    specs["A4"] = (
        a4_out,
        base
        + (
            "--output-root",
            str(a4_out),
            "--visual-encoder",
            "clip",
            "--clip-model-name",
            clip_model,
            "--clip-return-sequence",
            "--visual-fuser",
            "select_speech",
            "--fusion-location",
            "decoder_prefix",
            "--decoder-prompt-adapter",
            "resampler",
            "--decoder-prompt-len",
            "16",
            "--loss-rank-shuffle",
            "--loss-rank-weight",
            "0.1",
            "--loss-rank-margin",
            "0.2",
            "--lr",
            "1e-3",
        ),
    )

    a5_out = out("A5", "A5_blip2_qformer_decoder_prompt")
    specs["A5"] = (
        a5_out,
        base
        + (
            "--output-root",
            str(a5_out),
            "--visual-encoder",
            "clip",
            "--clip-model-name",
            clip_model,
            "--clip-return-sequence",
            "--visual-fuser",
            "select_speech",
            "--fusion-location",
            "decoder_prefix",
            "--decoder-prompt-adapter",
            "blip2_qformer",
            "--blip2-model-name",
            blip2_model,
            "--decoder-prompt-len",
            "16",
            "--lr",
            "1e-4",
        ),
    )

    a6_out = out("A6", "A6_decoder_prompt_lora")
    specs["A6"] = (
        a6_out,
        base
        + (
            "--output-root",
            str(a6_out),
            "--visual-encoder",
            "clip",
            "--clip-model-name",
            clip_model,
            "--clip-return-sequence",
            "--visual-fuser",
            "select_speech",
            "--fusion-location",
            "decoder_prefix",
            "--decoder-prompt-adapter",
            "resampler",
            "--decoder-prompt-len",
            "16",
            "--loss-rank-shuffle",
            "--loss-rank-weight",
            "0.1",
            "--loss-rank-margin",
            "0.2",
            "--enable-decoder-lora",
            "--lora-rank",
            "4",
            "--lora-alpha",
            "16",
            "--lora-last-n-layers",
            "4",
            "--lr",
            "2e-5",
        ),
    )

    result: List[JobSpec] = []
    for name in jobs:
        if name not in specs:
            raise ValueError(f"Unsupported job {name!r}; supported: {', '.join(sorted(specs))}")
        exp_out, args = specs[name]
        checkpoint = exp_out / "checkpoints" / "last.pt"
        full_args = args + ("--resume-from", str(checkpoint))
        result.append(JobSpec(name=name, gpu=gpu_map[name], output_root=exp_out, args=full_args))
    return result


def append_resume_log_path(log_dir: Path, job_name: str) -> Path:
    date_tag = time.strftime("%Y%m%d")
    return log_dir / f"{job_name}.monitor_resume.{date_tag}.log"


def start_job(
    *,
    job: JobSpec,
    repo_root: Path,
    python_exe: Path,
    log_dir: Path,
    dry_run: bool,
) -> Optional[int]:
    checkpoint_path = job.output_root / "checkpoints" / "last.pt"
    command = [str(python_exe), *job.args]
    log_path = append_resume_log_path(log_dir, job.name)
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = job.gpu
    env["DEVICE"] = "cuda"
    env["PYTHONNOUSERSITE"] = "1"
    env["CONDA_PREFIX"] = str(python_exe.parent.parent)
    env["PATH"] = f"{python_exe.parent}:{env.get('PATH', '')}"

    pretty = " ".join(shlex.quote(part) for part in command)
    log(f"{job.name}: starting on gpu={job.gpu}")
    log(f"{job.name}: resume checkpoint={checkpoint_path}")
    log(f"{job.name}: log={log_path}")
    log(f"{job.name}: command={pretty}")
    if dry_run:
        return None

    log_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as stream:
        stream.write(f"\n\n[{now()}] monitor starting {job.name} on GPU {job.gpu}\n".encode("utf-8"))
        proc = subprocess.Popen(
            command,
            cwd=str(repo_root),
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    log(f"{job.name}: started pid={proc.pid}")
    return proc.pid


def load_state(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data  # type: ignore[return-value]


def save_state(path: Path, state: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def should_restart(
    *,
    state: Dict[str, Dict[str, float]],
    job_name: str,
    min_restart_seconds: int,
    max_restarts: int,
) -> Tuple[bool, str]:
    record = state.setdefault(job_name, {"restart_count": 0, "last_restart_time": 0})
    restart_count = int(record.get("restart_count", 0))
    last_restart = float(record.get("last_restart_time", 0))
    if max_restarts > 0 and restart_count >= max_restarts:
        return False, f"max restarts reached ({restart_count}/{max_restarts})"
    age = time.time() - last_restart
    if last_restart > 0 and age < min_restart_seconds:
        return False, f"restart backoff active ({int(age)}s/{min_restart_seconds}s)"
    return True, "ok"


def mark_restart(state: Dict[str, Dict[str, float]], job_name: str) -> None:
    record = state.setdefault(job_name, {"restart_count": 0, "last_restart_time": 0})
    record["restart_count"] = int(record.get("restart_count", 0)) + 1
    record["last_restart_time"] = time.time()


def monitor_once(
    *,
    jobs: Sequence[JobSpec],
    repo_root: Path,
    python_exe: Path,
    log_dir: Path,
    state: Dict[str, Dict[str, float]],
    state_file: Path,
    target_epochs: int,
    dry_run: bool,
    require_checkpoint: bool,
    min_restart_seconds: int,
    max_restarts: int,
) -> None:
    for job in jobs:
        live = find_live_training_processes(job.output_root)
        if live:
            log(f"{job.name}: alive pid={','.join(map(str, live))}")
            continue

        checkpoint = job.output_root / "checkpoints" / "last.pt"
        if require_checkpoint and not checkpoint.is_file():
            log(f"{job.name}: skipped; missing checkpoint {checkpoint}")
            continue

        summary = load_checkpoint_summary(checkpoint, python_exe)
        if summary.get("error"):
            log(f"{job.name}: skipped; checkpoint inspect failed: {summary['error']}")
            continue

        if is_done(summary, target_epochs):
            log(f"{job.name}: done; checkpoint summary={summary}")
            continue

        allowed, reason = should_restart(
            state=state,
            job_name=job.name,
            min_restart_seconds=min_restart_seconds,
            max_restarts=max_restarts,
        )
        if not allowed:
            log(f"{job.name}: skipped; {reason}")
            continue

        start_job(job=job, repo_root=repo_root, python_exe=python_exe, log_dir=log_dir, dry_run=dry_run)
        if not dry_run:
            mark_restart(state, job.name)
            save_state(state_file, state)


def handle_stop(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=DEFAULT_REPO_ROOT)
    parser.add_argument("--env-prefix", default=DEFAULT_ENV_PREFIX)
    parser.add_argument("--suite-root", default=DEFAULT_SUITE_ROOT)
    parser.add_argument("--data-split", default=DEFAULT_DATA_SPLIT)
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--blip2-model", default=DEFAULT_BLIP2_MODEL)
    parser.add_argument(
        "--jobs",
        default="A2,A5,A6",
        help="Comma-separated jobs to monitor. Supported: A1,A2,A3,A4,A5,A6.",
    )
    parser.add_argument(
        "--gpu-map",
        action="append",
        default=[],
        help="Override GPU assignment, e.g. --gpu-map A2:1 --gpu-map A6:0.",
    )
    parser.add_argument(
        "--output-override",
        action="append",
        default=[],
        help=(
            "Override one job output root, e.g. "
            "--output-override A1=outputs/flickr8k_decoder_prompt_large_v3_A1_rerun_.../A1_blank_decoder_prefix_k16"
        ),
    )
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--interval", type=int, default=300, help="Seconds between checks.")
    parser.add_argument("--min-restart-seconds", type=int, default=600)
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=0,
        help="Maximum restarts per job; 0 means unlimited.",
    )
    parser.add_argument("--state-file", default="logs/monitor_flickr8k_decoder_prompt_state.json")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--once", action="store_true", help="Check once and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without starting jobs.")
    parser.add_argument(
        "--allow-start-without-checkpoint",
        action="store_true",
        help="Allow starting a job from scratch if last.pt is missing.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    env_prefix = Path(args.env_prefix).resolve()
    python_exe = env_prefix / "bin" / "python"
    if not python_exe.is_file():
        raise FileNotFoundError(f"Python executable not found: {python_exe}")

    names = csv_names(args.jobs)
    gpu_map = {
        "A1": "3",
        "A2": "1",
        "A3": "5",
        "A4": "0",
        "A5": "5",
        "A6": "0",
    }
    gpu_map.update(parse_gpu_map(args.gpu_map))

    for name in names:
        if name not in gpu_map:
            raise ValueError(f"No GPU configured for {name}; use --gpu-map {name}:<gpu>")

    output_overrides = parse_key_value_overrides(args.output_override)
    suite_root = Path(args.suite_root)
    data_split = Path(args.data_split)
    train_manifest = str(data_split / "train_manifest.jsonl")
    val_manifest = str(data_split / "val_manifest.jsonl")
    jobs = build_jobs(
        repo_root=repo_root,
        suite_root=suite_root,
        jobs=names,
        gpu_map=gpu_map,
        output_overrides=output_overrides,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        whisper_model=args.whisper_model,
        clip_model=args.clip_model,
        blip2_model=args.blip2_model,
        target_epochs=args.target_epochs,
        batch_size=args.batch_size,
    )

    state_file = resolve_path(args.state_file, repo_root)
    log_dir = resolve_path(args.log_dir, repo_root)
    state = load_state(state_file)

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    log(f"monitor started; jobs={','.join(job.name for job in jobs)} interval={args.interval}s")
    log(f"repo_root={repo_root}")
    log(f"state_file={state_file}")

    try:
        while True:
            monitor_once(
                jobs=jobs,
                repo_root=repo_root,
                python_exe=python_exe,
                log_dir=log_dir,
                state=state,
                state_file=state_file,
                target_epochs=args.target_epochs,
                dry_run=args.dry_run,
                require_checkpoint=not args.allow_start_without_checkpoint,
                min_restart_seconds=args.min_restart_seconds,
                max_restarts=args.max_restarts,
            )
            if args.once:
                break
            time.sleep(max(10, args.interval))
    except KeyboardInterrupt as exc:
        log(f"monitor stopped: {exc}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
