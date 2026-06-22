from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import save_file

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face pytorch_model.bin checkpoint into model.safetensors."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing pytorch_model.bin.",
    )
    parser.add_argument(
        "--bin-name",
        type=str,
        default="pytorch_model.bin",
        help="Source pickle checkpoint filename.",
    )
    parser.add_argument(
        "--safe-name",
        type=str,
        default="model.safetensors",
        help="Output safetensors filename.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the destination if it already exists.",
    )
    return parser.parse_args()


def unwrap_state_dict(payload: object) -> Dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(payload)!r}")

    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload  # already a plain state_dict

    for key in ("state_dict", "model_state_dict"):
        nested = payload.get(key)
        if isinstance(nested, dict) and nested and all(torch.is_tensor(value) for value in nested.values()):
            return nested

    tensor_keys = [key for key, value in payload.items() if torch.is_tensor(value)]
    if tensor_keys and len(tensor_keys) == len(payload):
        return payload

    raise ValueError(
        "Could not extract a plain tensor state_dict from checkpoint. "
        "Expected either a tensor dict or a nested state_dict/model_state_dict."
    )


def make_state_dict_contiguous(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    packed: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        packed[key] = value.contiguous() if torch.is_tensor(value) else value
    return packed


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    bin_path = model_dir / args.bin_name
    safe_path = model_dir / args.safe_name

    if not bin_path.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {bin_path}")
    if safe_path.exists() and not args.overwrite:
        raise FileExistsError(f"Destination already exists: {safe_path}. Pass --overwrite to replace it.")

    checkpoint = torch.load(bin_path, map_location="cpu")
    state_dict = make_state_dict_contiguous(unwrap_state_dict(checkpoint))
    save_file(state_dict, str(safe_path))
    print(f"[DONE] source={bin_path}")
    print(f"[DONE] output={safe_path}")
    print(f"[DONE] tensors={len(state_dict)}")


if __name__ == "__main__":
    main()
