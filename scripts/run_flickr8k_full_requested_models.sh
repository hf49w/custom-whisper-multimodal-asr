#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/.cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
IMAGES_ROOT="${IMAGES_ROOT:-$DATA_ROOT/flickr8k/images}"
AUDIO_ROOT="${AUDIO_ROOT:-$DATA_ROOT/flickr8k/audio}"
CAPTIONS_PATH="${CAPTIONS_PATH:-$DATA_ROOT/flickr8k/captions/captions.txt}"
PREPARED_ROOT="${PREPARED_ROOT:-$DATA_ROOT/flickr8k/prepared}"
FULL_MANIFEST_PATH="${FULL_MANIFEST_PATH:-$PREPARED_ROOT/manifest.jsonl}"
WHISPER_DOWNLOAD_ROOT="${WHISPER_DOWNLOAD_ROOT:-$DATA_ROOT/models/whisper}"
OFFLINE="${OFFLINE:-0}"

SPLIT_SEED="${SPLIT_SEED:-42}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
VAL_RATIO_TAG="${VAL_RATIO_TAG:-10}"
TEST_RATIO_TAG="${TEST_RATIO_TAG:-10}"

WHISPER_MODEL="${WHISPER_MODEL:-medium.en}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$BATCH_SIZE}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-$NUM_WORKERS}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
EVAL_PREFETCH_FACTOR="${EVAL_PREFETCH_FACTOR:-$PREFETCH_FACTOR}"
PIN_MEMORY="${PIN_MEMORY:-1}"
EVAL_PIN_MEMORY="${EVAL_PIN_MEMORY:-$PIN_MEMORY}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"
EVAL_PERSISTENT_WORKERS="${EVAL_PERSISTENT_WORKERS:-$PERSISTENT_WORKERS}"
LOG_EVERY="${LOG_EVERY:-10}"
EVAL_LOG_EVERY="${EVAL_LOG_EVERY:-20}"
SAVE_EVERY="${SAVE_EVERY:-1}"
SAVE_EVERY_BATCHES="${SAVE_EVERY_BATCHES:-100}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
NUM_GMLP_LAYERS="${NUM_GMLP_LAYERS:-1}"
P_SPEECH="${P_SPEECH:-0.5}"
DIM_SPEECH_INTER="${DIM_SPEECH_INTER:-128}"
DIM_VISUAL_INTER="${DIM_VISUAL_INTER:-128}"
CLIP_MODEL_NAME="${CLIP_MODEL_NAME:-openai/clip-vit-base-patch32}"
ATTN_NUM_HEADS="${ATTN_NUM_HEADS:-8}"
ATTN_DROPOUT="${ATTN_DROPOUT:-0.1}"
ATTN_GATE_INIT="${ATTN_GATE_INIT:--4.0}"
ATTN_NUM_QUERIES="${ATTN_NUM_QUERIES:-8}"
CKPT_NAME="${CKPT_NAME:-best_val_loss.pt}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
ENABLE_SPECAUG="${ENABLE_SPECAUG:-0}"
ALLOW_RELOCATED_PATHS="${ALLOW_RELOCATED_PATHS:-0}"
CLIP_PARALLEL_GPUS="${CLIP_PARALLEL_GPUS:-}"
PARALLEL_GPUS="${PARALLEL_GPUS:-}"

SPLIT_ROOT_DEFAULT="$PREPARED_ROOT/splits/by_image_id_seed${SPLIT_SEED}_val${VAL_RATIO_TAG}_test${TEST_RATIO_TAG}"
SPLIT_ROOT="${SPLIT_ROOT:-$SPLIT_ROOT_DEFAULT}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$SPLIT_ROOT/train_manifest.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$SPLIT_ROOT/val_manifest.jsonl}"
TEST_MANIFEST="${TEST_MANIFEST:-$SPLIT_ROOT/test_manifest.jsonl}"

SUITE_TAG_DEFAULT="flickr8k_full_seed${SPLIT_SEED}_val${VAL_RATIO_TAG}_test${TEST_RATIO_TAG}_ep${EPOCHS}"
SUITE_TAG="${SUITE_TAG:-$SUITE_TAG_DEFAULT}"
SUITE_ROOT="${SUITE_ROOT:-$REPO_ROOT/outputs/$SUITE_TAG}"
SUMMARY_TSV="${SUMMARY_TSV:-$SUITE_ROOT/summary.tsv}"

mkdir -p "$SUITE_ROOT"
mkdir -p "$PREPARED_ROOT" "$WHISPER_DOWNLOAD_ROOT" "$CACHE_ROOT"

run_python() {
  if [[ "$OFFLINE" == "1" ]]; then
    PYTHONUNBUFFERED="$PYTHONUNBUFFERED" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PYTHON_BIN" "$@"
  else
    PYTHONUNBUFFERED="$PYTHONUNBUFFERED" "$PYTHON_BIN" "$@"
  fi
}

if [[ ! -f "$FULL_MANIFEST_PATH" ]]; then
  echo "[INFO] Prepared Flickr8k manifest missing. Building it first."
  run_python "$REPO_ROOT/scripts/prepare_flickr8k_for_custom_whisper.py" \
    --images-root "$IMAGES_ROOT" \
    --audio-root "$AUDIO_ROOT" \
    --captions-path "$CAPTIONS_PATH" \
    --output-root "$PREPARED_ROOT"
fi

if [[ ! -f "$TRAIN_MANIFEST" || ! -f "$VAL_MANIFEST" || ! -f "$TEST_MANIFEST" ]]; then
  echo "[INFO] Building Flickr8k full train/val/test split by image_id."
  run_python "$REPO_ROOT/scripts/split_manifest_train_val_test.py" \
    --manifest-path "$FULL_MANIFEST_PATH" \
    --output-root "$SPLIT_ROOT" \
    --val-ratio "$VAL_RATIO" \
    --test-ratio "$TEST_RATIO" \
    --seed "$SPLIT_SEED" \
    --group-by-field image_id
fi

cat > "$SUMMARY_TSV" <<'EOF'
encoder	fuser	experiment_root	checkpoint	train_manifest	val_manifest	test_manifest
EOF

run_one_combo() {
  local encoder="$1"
  local fuser="$2"
  local resnet_depth="${3:-18}"
  local combo_name="${encoder}__${fuser}"
  local experiment_root="$SUITE_ROOT/$combo_name"
  local checkpoint_path="$experiment_root/model/checkpoints/$CKPT_NAME"

  echo "[RUN] encoder=$encoder fuser=$fuser resnet_depth=$resnet_depth force_retrain=$FORCE_RETRAIN"
  IMAGES_ROOT="$IMAGES_ROOT" \
  AUDIO_ROOT="$AUDIO_ROOT" \
  CAPTIONS_PATH="$CAPTIONS_PATH" \
  PREPARED_ROOT="$PREPARED_ROOT" \
  MANIFEST_PATH="$FULL_MANIFEST_PATH" \
  TRAIN_MANIFEST="$TRAIN_MANIFEST" \
  VAL_MANIFEST="$VAL_MANIFEST" \
  TEST_MANIFEST="$TEST_MANIFEST" \
  SPLIT_SEED="$SPLIT_SEED" \
  TEST_RATIO="$TEST_RATIO" \
  TEST_RATIO_TAG="$TEST_RATIO_TAG" \
  WHISPER_MODEL="$WHISPER_MODEL" \
  VISUAL_ENCODER="$encoder" \
  VISUAL_FUSER="$fuser" \
  EPOCHS="$EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
  LR="$LR" \
  WEIGHT_DECAY="$WEIGHT_DECAY" \
  NUM_WORKERS="$NUM_WORKERS" \
  EVAL_NUM_WORKERS="$EVAL_NUM_WORKERS" \
  PREFETCH_FACTOR="$PREFETCH_FACTOR" \
  EVAL_PREFETCH_FACTOR="$EVAL_PREFETCH_FACTOR" \
  PIN_MEMORY="$PIN_MEMORY" \
  EVAL_PIN_MEMORY="$EVAL_PIN_MEMORY" \
  PERSISTENT_WORKERS="$PERSISTENT_WORKERS" \
  EVAL_PERSISTENT_WORKERS="$EVAL_PERSISTENT_WORKERS" \
  DEVICE="${DEVICE:-}" \
  EVAL_DEVICE="${EVAL_DEVICE:-$DEVICE}" \
  LOG_EVERY="$LOG_EVERY" \
  EVAL_LOG_EVERY="$EVAL_LOG_EVERY" \
  SAVE_EVERY="$SAVE_EVERY" \
  SAVE_EVERY_BATCHES="$SAVE_EVERY_BATCHES" \
  IMAGE_SIZE="$IMAGE_SIZE" \
  NUM_GMLP_LAYERS="$NUM_GMLP_LAYERS" \
  NUM_RESNET_LAYERS="$resnet_depth" \
  P_SPEECH="$P_SPEECH" \
  DIM_SPEECH_INTER="$DIM_SPEECH_INTER" \
  DIM_VISUAL_INTER="$DIM_VISUAL_INTER" \
  CLIP_MODEL_NAME="$CLIP_MODEL_NAME" \
  CKPT_NAME="$CKPT_NAME" \
  WHISPER_DOWNLOAD_ROOT="$WHISPER_DOWNLOAD_ROOT" \
  ATTN_NUM_HEADS="$ATTN_NUM_HEADS" \
  ATTN_DROPOUT="$ATTN_DROPOUT" \
  ATTN_GATE_INIT="$ATTN_GATE_INIT" \
  ATTN_NUM_QUERIES="$ATTN_NUM_QUERIES" \
  FORCE_RETRAIN="$FORCE_RETRAIN" \
  ENABLE_SPECAUG="$ENABLE_SPECAUG" \
  ALLOW_RELOCATED_PATHS="$ALLOW_RELOCATED_PATHS" \
  OFFLINE="$OFFLINE" \
  SKIP_EVAL_TRAIN=1 \
  SKIP_EVAL_VAL=0 \
  SKIP_EVAL_TEST=0 \
  EXPERIMENT_ROOT="$experiment_root" \
  bash "$REPO_ROOT/scripts/run_flickr8k_custom_whisper_fuser.sh"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$encoder" \
    "$fuser" \
    "$experiment_root" \
    "$checkpoint_path" \
    "$TRAIN_MANIFEST" \
    "$VAL_MANIFEST" \
    "$TEST_MANIFEST" >> "$SUMMARY_TSV"
}

run_one_combo_on_gpu() {
  local gpu="$1"
  local encoder="$2"
  local fuser="$3"
  local resnet_depth="${4:-18}"
  local combo_name="${encoder}__${fuser}"
  local log_dir="$SUITE_ROOT/logs"
  local log_path="$log_dir/$combo_name.log"
  mkdir -p "$log_dir"
  echo "[RUN PARALLEL] encoder=$encoder fuser=$fuser gpu=$gpu log=$log_path" >&2
  (
    export DEVICE="cuda:$gpu"
    export EVAL_DEVICE="cuda:$gpu"
    run_one_combo "$encoder" "$fuser" "$resnet_depth"
  ) >"$log_path" 2>&1 &
  clip_pids+=("$!")
}

run_jobs_in_batches() {
  local gpu_csv="$1"
  shift
  local -a jobs=("$@")
  local -a gpus=()
  IFS=',' read -r -a gpus <<< "$gpu_csv"
  if [[ "${#gpus[@]}" -eq 0 ]]; then
    echo "[ERROR] no GPUs provided to run_jobs_in_batches" >&2
    exit 1
  fi
  local job_index=0
  while [[ "$job_index" -lt "${#jobs[@]}" ]]; do
    clip_pids=()
    local batch_size="${#gpus[@]}"
    local batch_end=$(( job_index + batch_size ))
    if [[ "$batch_end" -gt "${#jobs[@]}" ]]; then
      batch_end="${#jobs[@]}"
    fi
    local gpu_index=0
    local current_index="$job_index"
    while [[ "$current_index" -lt "$batch_end" ]]; do
      IFS='|' read -r encoder fuser resnet_depth <<< "${jobs[$current_index]}"
      run_one_combo_on_gpu "${gpus[$gpu_index]}" "$encoder" "$fuser" "$resnet_depth"
      gpu_index=$(( gpu_index + 1 ))
      current_index=$(( current_index + 1 ))
    done
    batch_failed=0
    for clip_pid in "${clip_pids[@]}"; do
      if ! wait "$clip_pid"; then
        batch_failed=1
      fi
    done
    if [[ "$batch_failed" == "1" ]]; then
      echo "[ERROR] One or more batched jobs failed. Check $SUITE_ROOT/logs/*.log" >&2
      exit 1
    fi
    job_index="$batch_end"
  done
}

all_jobs=(
  "resnet50|proj_concat_proj|50"
  "resnet_gmlp|concat_temp|18"
  "clip|cross_attn_gate|18"
  "clip|attn_prefix|18"
  "clip|gated_seq_concat|18"
)

if [[ -n "$PARALLEL_GPUS" ]]; then
  run_jobs_in_batches "$PARALLEL_GPUS" "${all_jobs[@]}"
elif [[ -n "$CLIP_PARALLEL_GPUS" ]]; then
  run_one_combo resnet50 proj_concat_proj 50
  run_one_combo resnet_gmlp concat_temp 18
  IFS=',' read -r -a clip_gpus <<< "$CLIP_PARALLEL_GPUS"
  if [[ "${#clip_gpus[@]}" -lt 3 ]]; then
    echo "[ERROR] CLIP_PARALLEL_GPUS requires at least 3 comma-separated GPU ids, got: $CLIP_PARALLEL_GPUS" >&2
    exit 1
  fi
  clip_pids=()
  run_one_combo_on_gpu "${clip_gpus[0]}" clip cross_attn_gate 18
  run_one_combo_on_gpu "${clip_gpus[1]}" clip attn_prefix 18
  run_one_combo_on_gpu "${clip_gpus[2]}" clip gated_seq_concat 18
  clip_failed=0
  for clip_pid in "${clip_pids[@]}"; do
    if ! wait "$clip_pid"; then
      clip_failed=1
    fi
  done
  if [[ "$clip_failed" == "1" ]]; then
    echo "[ERROR] One or more parallel CLIP jobs failed. Check $SUITE_ROOT/logs/*.log" >&2
    exit 1
  fi
else
  run_one_combo resnet50 proj_concat_proj 50
  run_one_combo resnet_gmlp concat_temp 18
  run_one_combo clip cross_attn_gate 18
  run_one_combo clip attn_prefix 18
  run_one_combo clip gated_seq_concat 18
fi

echo "[DONE] train_manifest=$TRAIN_MANIFEST"
echo "[DONE] val_manifest=$VAL_MANIFEST"
echo "[DONE] test_manifest=$TEST_MANIFEST"
echo "[DONE] summary=$SUMMARY_TSV"
