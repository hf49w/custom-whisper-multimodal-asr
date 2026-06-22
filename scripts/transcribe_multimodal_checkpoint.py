from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import custom_whisper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe one audio file with a trained AudioImageWhisper checkpoint."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--image-path", default="")
    parser.add_argument(
        "--whisper-model",
        default="",
        help="Local Whisper .pt path. Overrides the path stored in the checkpoint.",
    )
    parser.add_argument(
        "--clip-model-name",
        default="",
        help="Local CLIP model directory. Overrides the path stored in the checkpoint.",
    )
    parser.add_argument("--device", default="", help="For example cuda, cuda:0, or cpu.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--task", choices=("transcribe", "translate"), default="transcribe")
    parser.add_argument("--initial-prompt", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Do not pass the image. Useful as an ablation baseline.",
    )
    parser.add_argument("--zero-prefix-at-eval", action="store_true")
    parser.add_argument("--use-trained-blank-prefix-at-eval", action="store_true")
    parser.add_argument(
        "--blank-prefix-at-eval",
        dest="zero_prefix_at_eval",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(no_download=True)
    parser.add_argument("--no-download", dest="no_download", action="store_true")
    parser.add_argument("--allow-download", dest="no_download", action="store_false")
    return parser.parse_args()


def require_file(raw_path: str, label: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def resolve_model_path(override: str, stored: str, label: str) -> str:
    candidate = override or stored
    if not candidate:
        raise ValueError(f"{label} is missing; provide its command-line override.")

    path = Path(candidate).expanduser()
    if path.exists():
        return str(path.resolve())
    if override:
        raise FileNotFoundError(f"{label} not found: {path}")
    raise FileNotFoundError(
        f"{label} stored in the checkpoint does not exist locally: {candidate}\n"
        f"Provide a local path with --{'whisper-model' if label == 'Whisper model' else 'clip-model-name'}."
    )


def rebuild_model(
    checkpoint: Dict[str, Any],
    *,
    device: torch.device,
    whisper_model: str,
    clip_model_name: str,
    no_download: bool = True,
) -> custom_whisper.AudioImageWhisper:
    config = checkpoint["train_config"]
    special_token_count = config.get(
        "resolved_decoder_prompt_special_tokens",
        config.get("decoder_prompt_special_tokens"),
    )
    if special_token_count is not None and int(special_token_count) < 0:
        special_token_count = None
    model = custom_whisper.load_audio_image_model(
        whisper_model,
        device=device,
        visual_encoder=config["visual_encoder"],
        feature_fuser=config["visual_fuser"],
        visual_pretrained=config["visual_pretrained"],
        image_size=config["image_size"],
        clip_model_name=clip_model_name,
        clip_return_sequence=config["clip_return_sequence"],
        num_gmlp_layers=config["num_gmlp_layers"],
        num_resnet_layers=config["num_resnet_layers"],
        p_speech=config["p_speech"],
        use_residual=config["use_residual"],
        dim_speech_inter=config["dim_speech_inter"],
        dim_visual_inter=config["dim_visual_inter"],
        use_layer_norm=config["use_layer_norm"],
        attn_num_heads=config.get("attn_num_heads", 8),
        attn_dropout=config.get("attn_dropout", 0.1),
        attn_gate_init=config.get("attn_gate_init", -4.0),
        attn_num_queries=config.get("attn_num_queries", 8),
        fusion_location=config.get("fusion_location", "encoder_memory"),
        decoder_prompt_adapter=config.get("decoder_prompt_adapter", "none"),
        decoder_prompt_len=config.get("decoder_prompt_len", 16),
        decoder_prompt_heads=config.get("decoder_prompt_heads", 8),
        decoder_prompt_dropout=config.get("decoder_prompt_dropout", 0.1),
        decoder_prompt_insert=config.get("decoder_prompt_insert", "before_tokens"),
        decoder_prompt_special_tokens=special_token_count,
        decoder_prompt_missing=config.get("decoder_prompt_missing", "audio_only"),
        blip2_model_name=config.get("blip2_model_name", ""),
        freeze_whisper=True,
        freeze_visual_encoder=True,
        visual_local_files_only=no_download,
        enable_decoder_lora=config.get("enable_decoder_lora", False),
        lora_rank=config.get("lora_rank", 4),
        lora_alpha=config.get("lora_alpha", 16.0),
        lora_dropout=config.get("lora_dropout", 0.05),
        lora_last_n_layers=config.get("lora_last_n_layers", 4),
        lora_targets=config.get("lora_targets", "self_attn_qv,cross_attn_qv,mlp"),
    )
    if "lightweight_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["lightweight_state_dict"], strict=False)
    elif "feature_fuser_state_dict" in checkpoint:
        model.feature_fuser.load_state_dict(checkpoint["feature_fuser_state_dict"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if sum([args.audio_only, args.zero_prefix_at_eval, args.use_trained_blank_prefix_at_eval]) > 1:
        raise ValueError("Choose only one audio/prefix diagnostic mode")
    checkpoint_path = require_file(args.checkpoint_path, "Checkpoint")
    audio_path = require_file(args.audio_path, "Audio")
    image_path = require_file(args.image_path, "Image") if args.image_path else None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("train_config")
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint has no train_config: {checkpoint_path}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    whisper_model = resolve_model_path(
        args.whisper_model,
        str(config.get("whisper_model", "")),
        "Whisper model",
    )
    clip_model_name = str(config.get("clip_model_name", ""))
    if config.get("visual_encoder") == "clip":
        clip_model_name = resolve_model_path(
            args.clip_model_name,
            clip_model_name,
            "CLIP model",
        )

    model = rebuild_model(
        checkpoint,
        device=device,
        whisper_model=whisper_model,
        clip_model_name=clip_model_name,
        no_download=args.no_download,
    )
    if args.zero_prefix_at_eval and model.fusion_location != "decoder_prefix":
        raise ValueError("--zero-prefix-at-eval requires fusion_location='decoder_prefix'")
    if (
        args.use_trained_blank_prefix_at_eval
        and (
            model.fusion_location != "decoder_prefix"
            or model.decoder_prompt_adapter_name != "blank_prefix"
        )
    ):
        raise ValueError(
            "--use-trained-blank-prefix-at-eval requires decoder_prompt_adapter='blank_prefix'"
        )
    transcribe_options: Dict[str, Any] = {
        "language": args.language,
        "task": args.task,
        "fp16": device.type == "cuda",
        "verbose": None,
    }
    if args.initial_prompt:
        transcribe_options["initial_prompt"] = args.initial_prompt
    no_image_mode = (
        args.audio_only
        or args.zero_prefix_at_eval
        or args.use_trained_blank_prefix_at_eval
        or model.decoder_prompt_adapter_name == "blank_prefix"
    )
    if not no_image_mode:
        if image_path is None:
            raise ValueError("--image-path is required unless audio-only/blank-prefix mode is used")
    if not no_image_mode and image_path is not None:
        transcribe_options["image"] = str(image_path)
    override = (
        "disabled"
        if args.audio_only
        else "zero"
        if args.zero_prefix_at_eval
        else "trained_blank"
        if args.use_trained_blank_prefix_at_eval
        else None
    )
    with model.use_decoder_prefix_override(override):
        result = model.transcribe(str(audio_path), **transcribe_options)
    payload = {
        "text": str(result.get("text", "")).strip(),
        "audio_path": str(audio_path),
        "image_path": "" if image_path is None or no_image_mode else str(image_path),
        "audio_only": bool(args.audio_only),
        "checkpoint_path": str(checkpoint_path),
        "whisper_model": whisper_model,
        "clip_model_name": clip_model_name,
        "visual_fuser": config["visual_fuser"],
        "device": str(device),
    }
    print(payload["text"])
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[DONE] output={output_path}")


if __name__ == "__main__":
    main()
