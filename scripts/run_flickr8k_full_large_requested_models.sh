#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA2_ROOT="${DATA2_ROOT:-/DATA_2/guest/custom-whisper}"
DATA2_TMP="${DATA2_TMP:-/DATA_2/guest/tmp}"
mkdir -p "$DATA2_TMP" "$DATA2_ROOT/data/models/whisper" "$DATA2_ROOT/.cache"

export TMPDIR="${TMPDIR:-$DATA2_TMP}"
export TMP="${TMP:-$DATA2_TMP}"
export TEMP="${TEMP:-$DATA2_TMP}"
export TORCH_HOME="${TORCH_HOME:-$DATA2_ROOT/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA2_ROOT/.cache/xdg}"
export HF_HOME="${HF_HOME:-$DATA2_ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

WHISPER_LARGE_PATH="${WHISPER_LARGE_PATH:-$DATA2_ROOT/data/models/whisper/large-v3.pt}"
if [[ ! -f "$WHISPER_LARGE_PATH" ]]; then
  cat >&2 <<EOF
[ERROR] Whisper large checkpoint is missing:
  $WHISPER_LARGE_PATH

Download it locally and upload it to that path, or set WHISPER_LARGE_PATH
to an existing local checkpoint file.
EOF
  exit 2
fi

export DATA_ROOT="${DATA_ROOT:-$DATA2_ROOT/data}"
export CACHE_ROOT="${CACHE_ROOT:-$DATA2_ROOT/.cache}"
export WHISPER_DOWNLOAD_ROOT="${WHISPER_DOWNLOAD_ROOT:-$DATA2_ROOT/data/models/whisper}"
export WHISPER_MODEL="${WHISPER_MODEL:-$WHISPER_LARGE_PATH}"
export CLIP_MODEL_NAME="${CLIP_MODEL_NAME:-$DATA2_ROOT/data/models/clip/clip-vit-base-patch32}"
export OFFLINE="${OFFLINE:-1}"
export SPLIT_SEED="${SPLIT_SEED:-42}"
export VAL_RATIO="${VAL_RATIO:-0.1}"
export TEST_RATIO="${TEST_RATIO:-0.1}"
export VAL_RATIO_TAG="${VAL_RATIO_TAG:-10}"
export TEST_RATIO_TAG="${TEST_RATIO_TAG:-10}"
export EPOCHS="${EPOCHS:-50}"
export SUITE_TAG="${SUITE_TAG:-flickr8k_full_large_seed${SPLIT_SEED}_val${VAL_RATIO_TAG}_test${TEST_RATIO_TAG}_ep${EPOCHS}}"
export SUITE_ROOT="${SUITE_ROOT:-$REPO_ROOT/outputs/$SUITE_TAG}"
export CKPT_NAME="${CKPT_NAME:-best_val_loss.pt}"
export FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
export ALLOW_RELOCATED_PATHS="${ALLOW_RELOCATED_PATHS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
export PIN_MEMORY="${PIN_MEMORY:-1}"
export EVAL_PIN_MEMORY="${EVAL_PIN_MEMORY:-1}"
export PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"
export EVAL_PERSISTENT_WORKERS="${EVAL_PERSISTENT_WORKERS:-1}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export EVAL_PREFETCH_FACTOR="${EVAL_PREFETCH_FACTOR:-2}"
export SAVE_EVERY="${SAVE_EVERY:-1}"
export SAVE_EVERY_BATCHES="${SAVE_EVERY_BATCHES:-100}"

# Avoid GPU3 by default. GPU0 may already be used by the previous medium run,
# so large jobs start on 1/2/4/5 unless PARALLEL_GPUS is explicitly set.
export PARALLEL_GPUS="${PARALLEL_GPUS:-1,2,4,5}"

exec bash "$REPO_ROOT/scripts/run_flickr8k_full_requested_models.sh"
