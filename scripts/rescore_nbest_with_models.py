"""Teacher-forcing rescore n-best candidates with one or more ASR models.

This script does not decode new hypotheses and does not train anything. It
adds per-candidate normalized teacher-forcing logprob fields such as
``A0_logprob`` or ``A2_logprob`` to an existing n-best JSONL file.

By default models are loaded sequentially: one model is loaded, all candidates
are rescored, intermediate JSONL is written, then the model is released before
the next model is loaded. This is intentionally slower than keeping every model
resident, but avoids OOM when rescoring large-v3 checkpoints on a 24GB GPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import custom_whisper
from a9_candidate_utils import candidate_text, read_jsonl, resolve_cross_platform_path
from dump_nbest_candidates import teacher_forced_token_logprobs
from eval_decode_only_oracle_rerank import load_eval_model, model_needs_images
from visspeech_custom_whisper_utils import build_tokenizer_and_prefix, normalize_eval_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Repeatable model spec. Format: NAME=checkpoint:/path/to/ckpt.pt "
            "or NAME=whisper:/path/to/model.pt"
        ),
    )
    parser.add_argument("--checkpoint", action="append", default=[], help="Repeatable shorthand NAME=PATH.")
    parser.add_argument("--whisper-model-spec", action="append", default=[], help="Repeatable shorthand NAME=PATH.")
    parser.add_argument("--device", default="")
    parser.add_argument("--visual-model-name", default="")
    parser.add_argument(
        "--base-whisper-model",
        default="",
        help="Override checkpoint train_config['whisper_model'] for portable inference.",
    )
    parser.add_argument(
        "--blip2-model-name",
        default="",
        help="Override checkpoint train_config['blip2_model_name'] for portable inference.",
    )
    parser.add_argument("--candidate-batch-size", type=int, default=32)
    parser.add_argument(
        "--load-all-models",
        action="store_true",
        help=(
            "Load all models at once. Faster when memory allows, but unsafe for "
            "multiple large-v3 checkpoints on 24GB GPUs. Default is sequential loading."
        ),
    )
    parser.add_argument(
        "--resume-output",
        action="store_true",
        help=(
            "If --output-jsonl already exists, read it as the starting point and "
            "skip model fields already present on every candidate."
        ),
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Recompute a model even if its *_logprob fields already exist.",
    )
    parser.set_defaults(no_download=True)
    parser.add_argument("--no-download", dest="no_download", action="store_true")
    parser.add_argument("--allow-download", dest="no_download", action="store_false")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def parse_model_specs(args: argparse.Namespace) -> List[Tuple[str, str, str]]:
    specs: List[Tuple[str, str, str]] = []
    for raw in args.model:
        name, value = raw.split("=", 1)
        if ":" not in value:
            raise ValueError(f"--model spec must include kind prefix checkpoint: or whisper:: {raw}")
        kind, path = value.split(":", 1)
        if kind not in {"checkpoint", "whisper"}:
            raise ValueError(f"Unsupported model kind in {raw}: {kind}")
        specs.append((name.strip(), kind, path.strip()))
    for raw in args.checkpoint:
        name, path = raw.split("=", 1)
        specs.append((name.strip(), "checkpoint", path.strip()))
    for raw in args.whisper_model_spec:
        name, path = raw.split("=", 1)
        specs.append((name.strip(), "whisper", path.strip()))
    if not specs:
        raise ValueError("Provide at least one --model, --checkpoint, or --whisper-model-spec")
    names = [name for name, _kind, _path in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate model names: {names}")
    return specs


def load_named_model(name: str, kind: str, path: str, args: argparse.Namespace, device: torch.device):
    namespace = SimpleNamespace(
        checkpoint_path=path if kind == "checkpoint" else "",
        whisper_model=path if kind == "whisper" else "",
        visual_model_name=args.visual_model_name,
        base_whisper_model=args.base_whisper_model,
        blip2_model_name=args.blip2_model_name,
        no_download=args.no_download,
    )
    model, source, is_checkpoint = load_eval_model(namespace, device)
    tokenizer, initial_tokens = build_tokenizer_and_prefix(model)
    use_images = model_needs_images(model, is_checkpoint=is_checkpoint, disable_images=False)
    return {
        "name": name,
        "kind": kind,
        "path": path,
        "source": source,
        "model": model,
        "tokenizer": tokenizer,
        "initial_tokens": initial_tokens,
        "use_images": use_images,
        "candidate_batch_size": int(args.candidate_batch_size),
    }


def encode_candidate_tokens(text: str, tokenizer: Any, model: Any, initial_tokens: Sequence[int]) -> List[int]:
    normalized = normalize_eval_text(text)
    token_ids = tokenizer.encode(" " + normalized) if normalized else []
    soft_prefix_len = (
        int(getattr(model, "decoder_prompt_len", 0))
        if (
            getattr(model, "fusion_location", None) == "decoder_prefix"
            and getattr(model, "visual_prompt_adapter", None) is not None
        )
        else 0
    )
    max_ctx = int(model.dims.n_text_ctx) - soft_prefix_len
    max_text_tokens = max(1, max_ctx - len(initial_tokens) - 1)
    return [int(token) for token in token_ids[:max_text_tokens]]


@torch.no_grad()
def audio_features_for_sample(model_info: Dict[str, Any], sample: Dict[str, Any], device: torch.device):
    model = model_info["model"]
    audio_path = str(sample.get("audio_path") or sample.get("wav_path"))
    if not audio_path:
        raise ValueError(f"Sample has no audio_path/wav_path: {sample.get('sample_id')}")
    audio = custom_whisper.pad_or_trim(custom_whisper.load_audio(audio_path))
    mel = custom_whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(device)
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    if device.type == "cuda":
        mel = mel.half()
    context = (
        model.use_visual_context(image=str(sample.get("image_path", "")))
        if model_info["use_images"] and hasattr(model, "use_visual_context")
        else nullcontext()
    )
    with context:
        features = model.embed_audio(mel)
    if features.dim() == 2:
        features = features.unsqueeze(0)
    return features


@torch.no_grad()
def batched_teacher_forced_token_logprobs(
    model: Any,
    audio_features: torch.Tensor,
    *,
    initial_tokens: Sequence[int],
    candidate_token_lists: Sequence[Sequence[int]],
    eot_token: int,
    device: torch.device,
    image_path: str = "",
    use_images: bool = False,
    batch_size: int = 32,
) -> List[List[float]]:
    if not candidate_token_lists:
        return []
    features = audio_features
    if features.dim() == 2:
        features = features.unsqueeze(0)
    features = features.to(device)
    outputs: List[List[float]] = []
    pad_token = int(eot_token)
    for start_index in range(0, len(candidate_token_lists), max(1, int(batch_size))):
        batch_tokens = candidate_token_lists[start_index : start_index + max(1, int(batch_size))]
        full_tokens = [
            list(initial_tokens) + [int(token) for token in candidate_tokens] + [int(eot_token)]
            for candidate_tokens in batch_tokens
        ]
        max_input_len = max(max(1, len(tokens) - 1) for tokens in full_tokens)
        input_tokens = torch.full(
            (len(full_tokens), max_input_len),
            pad_token,
            dtype=torch.long,
            device=device,
        )
        labels = torch.full_like(input_tokens, pad_token)
        sequence_lengths: List[int] = []
        for row_index, tokens in enumerate(full_tokens):
            if len(tokens) < 2:
                sequence_lengths.append(0)
                continue
            sequence_len = len(tokens) - 1
            sequence_lengths.append(sequence_len)
            input_tokens[row_index, :sequence_len] = torch.tensor(tokens[:-1], dtype=torch.long, device=device)
            labels[row_index, :sequence_len] = torch.tensor(tokens[1:], dtype=torch.long, device=device)

        batch_features = features
        if batch_features.shape[0] == 1 and len(full_tokens) > 1:
            batch_features = batch_features.repeat(len(full_tokens), 1, 1)
        elif batch_features.shape[0] != len(full_tokens):
            raise ValueError(
                f"Audio feature batch mismatch: features={batch_features.shape[0]} candidates={len(full_tokens)}"
            )

        if use_images and hasattr(model, "use_visual_context"):
            with model.use_visual_context(image=image_path):
                logits = model.logits(input_tokens, batch_features)
        else:
            logits = model.logits(input_tokens, batch_features)
        logprobs = F.log_softmax(logits.float(), dim=-1)
        gathered = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        for row_index, candidate_tokens in enumerate(batch_tokens):
            if sequence_lengths[row_index] == 0:
                outputs.append([])
                continue
            score_start = max(0, len(initial_tokens) - 1)
            score_end = min(sequence_lengths[row_index], score_start + len(candidate_tokens) + 1)
            outputs.append([float(value) for value in gathered[row_index, score_start:score_end].detach().cpu().tolist()])
    return outputs


def rescore_sample(sample: Dict[str, Any], model_info: Dict[str, Any], device: torch.device) -> None:
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    initial_tokens = model_info["initial_tokens"]
    features = audio_features_for_sample(model_info, sample, device)
    image_path = str(sample.get("image_path", ""))
    name = model_info["name"]
    candidates = sample.get("candidates", [])
    token_batches = [
        encode_candidate_tokens(candidate_text(candidate), tokenizer, model, initial_tokens)
        for candidate in candidates
    ]
    batch_logprobs = batched_teacher_forced_token_logprobs(
        model,
        features,
        initial_tokens=initial_tokens,
        candidate_token_lists=token_batches,
        eot_token=tokenizer.eot,
        device=device,
        image_path=image_path,
        use_images=bool(model_info["use_images"]),
        batch_size=int(model_info.get("candidate_batch_size", 32)),
    )
    for candidate, token_logprobs in zip(candidates, batch_logprobs):
        mean_logprob = float(sum(token_logprobs) / len(token_logprobs)) if token_logprobs else 0.0
        candidate[f"{name}_logprob"] = mean_logprob
        candidate[f"{name}_sum_logprob"] = float(sum(token_logprobs)) if token_logprobs else 0.0
        candidate[f"{name}_min_token_logprob"] = float(min(token_logprobs)) if token_logprobs else 0.0


def write_samples_jsonl(path: Path, samples: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def all_candidates_have_model_score(samples: Sequence[Dict[str, Any]], model_name: str) -> bool:
    field = f"{model_name}_logprob"
    for sample in samples:
        candidates = sample.get("candidates", [])
        if not candidates:
            continue
        for candidate in candidates:
            if field not in candidate:
                return False
    return True


def clone_samples(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cloned: List[Dict[str, Any]] = []
    for sample in samples:
        out_sample = dict(sample)
        out_sample["candidates"] = [dict(candidate) for candidate in sample.get("candidates", [])]
        cloned.append(out_sample)
    return cloned


def release_model(model_info: Dict[str, Any], device: torch.device) -> None:
    model = model_info.get("model")
    if model is not None and hasattr(model, "to"):
        try:
            model.to("cpu")
        except Exception:
            pass
    model_info.clear()
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def rescore_samples_for_model(
    samples: List[Dict[str, Any]],
    *,
    spec: Tuple[str, str, str],
    args: argparse.Namespace,
    device: torch.device,
    output_jsonl: Path,
) -> None:
    name, kind, path = spec
    if not args.no_skip_existing and all_candidates_have_model_score(samples, name):
        print(f"[RESCORE_SKIP] model={name} field={name}_logprob exists for all candidates")
        return

    print(f"[MODEL_LOAD] name={name} kind={kind} path={path}")
    model_info = load_named_model(name, kind, path, args, device)
    try:
        for index, sample in enumerate(samples, start=1):
            rescore_sample(sample, model_info, device)
            if index == 1 or index == len(samples) or index % max(1, args.log_every) == 0:
                print(f"[RESCORE] sample={index}/{len(samples)} model={name}")
        write_samples_jsonl(output_jsonl, samples)
        print(f"[MODEL_DONE] name={name} wrote={output_jsonl}")
    finally:
        release_model(model_info, device)


def main() -> None:
    args = parse_args()
    input_jsonl = resolve_cross_platform_path(args.input_jsonl)
    output_jsonl = resolve_cross_platform_path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    start_jsonl = output_jsonl if args.resume_output and output_jsonl.exists() and output_jsonl.stat().st_size > 0 else input_jsonl
    raw_samples = read_jsonl(start_jsonl)
    if not raw_samples:
        raise ValueError(f"No samples loaded from {start_jsonl}")
    samples = clone_samples(raw_samples)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    specs = parse_model_specs(args)

    if args.load_all_models:
        model_infos = [
            load_named_model(name, kind, path, args, device)
            for name, kind, path in specs
        ]
        try:
            for index, sample in enumerate(samples, start=1):
                for model_info in model_infos:
                    rescore_sample(sample, model_info, device)
                if index == 1 or index == len(samples) or index % max(1, args.log_every) == 0:
                    print(f"[RESCORE] sample={index}/{len(samples)} models={','.join(name for name, _kind, _path in specs)}")
            write_samples_jsonl(output_jsonl, samples)
        finally:
            for model_info in model_infos:
                release_model(model_info, device)
        return

    print(
        f"[RESCORE_MODE] sequential models={','.join(name for name, _kind, _path in specs)} "
        f"start={start_jsonl} output={output_jsonl}"
    )
    for spec in specs:
        rescore_samples_for_model(
            samples,
            spec=spec,
            args=args,
            device=device,
            output_jsonl=output_jsonl,
        )
    write_samples_jsonl(output_jsonl, samples)


if __name__ == "__main__":
    main()
