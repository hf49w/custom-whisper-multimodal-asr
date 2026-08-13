#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run A1 beam20 + validation-tuned post-rescore inference.

This is the deployment/inference pipeline for the official A1 beam20
post-rescore setup:

  score =
    0.5  * asr_mean_logprob
  + 0.01 * clip_zscore
  + 1.0  * mbr_score
  + 0.5  * A1_logprob
  + 0.05 * A5_logprob

A0/A2 teacher-forcing scores and length_score are intentionally not computed
because their validation-tuned weights are zero for the accepted beam20 run.

Required:
  --manifest PATH        JSONL manifest with audio_path/wav_path and image_path.
  --output-dir PATH      Output directory.

Optional:
  --gpu ID               CUDA_VISIBLE_DEVICES value. Default: 0.
  --python PATH          Python executable. Default: env PY or lab-252 env.
  --a1-ckpt PATH         A1 checkpoint.
  --a5-ckpt PATH         A5 checkpoint.
  --base-whisper-model PATH
                         Local Whisper large-v3 checkpoint used to rebuild A1/A5.
  --clip-model PATH      Local CLIP model directory.
  --blip2-model PATH     Local BLIP2 model directory for A5 if required.
  --batch-size N         Candidate teacher-forcing batch size. Default: 4.
  --max-samples N        Debug subset for dump_nbest_candidates.py.
  --reuse               Reuse existing intermediate JSONL when present.

Outputs:
  OUTPUT_DIR/a1_beam20_nbest.jsonl
  OUTPUT_DIR/a1_beam20_A1_A5_rescored.jsonl
  OUTPUT_DIR/final_rerank/predictions_best.jsonl
  OUTPUT_DIR/final_predictions.jsonl
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-}"
MANIFEST=""
OUT=""
GPU="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE=4
MAX_SAMPLES=0
REUSE=0

A1_CKPT="${A1_CKPT:-$ROOT/outputs/flickr8k_decoder_prompt_large_v3_A1_rerun_20260626_165442/A1_blank_decoder_prefix_k16/checkpoints/best_val_loss.pt}"
A5_CKPT="${A5_CKPT:-$ROOT/outputs/flickr8k_decoder_prompt_large_v3/A5_blip2_qformer_decoder_prompt/checkpoints/best_val_loss.pt}"
BASE_WHISPER_MODEL="${BASE_WHISPER_MODEL:-/DATA_2/guest/custom-whisper/data/models/whisper/large-v3.pt}"
CLIP_MODEL="${CLIP_MODEL:-/DATA_2/guest/custom-whisper/data/models/clip/clip-vit-base-patch32}"
BLIP2_MODEL="${BLIP2_MODEL:-/DATA_2/guest/custom-whisper/data/models/blip2/blip2-opt-2.7b}"

if [[ -z "$PY" ]]; then
  if [[ -x /DATA_4/guest/envs/custom-whisper-mm/bin/python ]]; then
    PY=/DATA_4/guest/envs/custom-whisper-mm/bin/python
  elif [[ -x /DATA_2/guest/envs/custom-whisper-mm/bin/python ]]; then
    PY=/DATA_2/guest/envs/custom-whisper-mm/bin/python
  else
    PY=python
  fi
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --output-dir) OUT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    --a1-ckpt) A1_CKPT="$2"; shift 2 ;;
    --a5-ckpt) A5_CKPT="$2"; shift 2 ;;
    --base-whisper-model) BASE_WHISPER_MODEL="$2"; shift 2 ;;
    --clip-model) CLIP_MODEL="$2"; shift 2 ;;
    --blip2-model) BLIP2_MODEL="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --reuse) REUSE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$MANIFEST" || -z "$OUT" ]]; then
  echo "[ERROR] --manifest and --output-dir are required" >&2
  usage >&2
  exit 2
fi

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] missing $label: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    echo "[ERROR] missing $label: $path" >&2
    exit 1
  fi
}

require_file "$MANIFEST" "manifest"
require_file "$A1_CKPT" "A1 checkpoint"
require_file "$A5_CKPT" "A5 checkpoint"
require_file "$BASE_WHISPER_MODEL" "base Whisper model"
require_dir "$CLIP_MODEL" "CLIP model"
if [[ -n "$BLIP2_MODEL" ]]; then
  require_dir "$BLIP2_MODEL" "BLIP2 model"
fi

mkdir -p "$OUT"
export PYTHONPATH="$ROOT/scripts:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

NBEST="$OUT/a1_beam20_nbest.jsonl"
RESCORED="$OUT/a1_beam20_A1_A5_rescored.jsonl"
RERANK_DIR="$OUT/final_rerank"

echo "[CONFIG] root=$ROOT"
echo "[CONFIG] python=$PY"
echo "[CONFIG] gpu=$GPU"
echo "[CONFIG] manifest=$MANIFEST"
echo "[CONFIG] output_dir=$OUT"
echo "[CONFIG] a1_ckpt=$A1_CKPT"
echo "[CONFIG] a5_ckpt=$A5_CKPT"
echo "[CONFIG] base_whisper_model=$BASE_WHISPER_MODEL"
echo "[CONFIG] clip_model=$CLIP_MODEL"
echo "[CONFIG] blip2_model=$BLIP2_MODEL"
echo "[CONFIG] score=0.5*ASR + 0.01*CLIP_z + 1.0*MBR + 0.5*A1_logprob + 0.05*A5_logprob"

DUMP_ARGS=(
  scripts/dump_nbest_candidates.py
  --checkpoint-path "$A1_CKPT"
  --manifest-path "$MANIFEST"
  --output-jsonl "$NBEST"
  --beam-size 20
  --n-best 20
  --device cuda
  --no-download
  --base-whisper-model "$BASE_WHISPER_MODEL"
  --visual-model-name "$CLIP_MODEL"
  --blip2-model-name "$BLIP2_MODEL"
  --log-every 20
)
if [[ "$MAX_SAMPLES" != "0" ]]; then
  DUMP_ARGS+=(--max-samples "$MAX_SAMPLES")
fi
if [[ "$REUSE" == "1" ]]; then
  DUMP_ARGS+=(--skip-if-exists)
fi

echo "[STEP] dump A1 beam20 candidates"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "${DUMP_ARGS[@]}"

echo "[STEP] rescore candidates with A1/A5 only"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/rescore_nbest_with_models.py \
  --input-jsonl "$NBEST" \
  --output-jsonl "$RESCORED" \
  --checkpoint A1="$A1_CKPT" \
  --checkpoint A5="$A5_CKPT" \
  --base-whisper-model "$BASE_WHISPER_MODEL" \
  --visual-model-name "$CLIP_MODEL" \
  --blip2-model-name "$BLIP2_MODEL" \
  --device cuda \
  --no-download \
  --candidate-batch-size "$BATCH_SIZE" \
  --resume-output \
  --log-every 20

echo "[STEP] fixed-weight post-rescore rerank"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/rerank_nbest_candidates.py \
  --input-jsonl "$RESCORED" \
  --output-dir "$RERANK_DIR" \
  --clip-model-name "$CLIP_MODEL" \
  --device cuda \
  --no-download \
  --a-values 0.5 \
  --b-values 0.01 \
  --c-values 1.0 \
  --d-values 0 \
  --extra-score-fields A1_logprob,A5_logprob \
  --extra-score-weight-specs A1_logprob=0.5,A5_logprob=0.05 \
  --log-every 20

cp "$RERANK_DIR/predictions_best.jsonl" "$OUT/final_predictions.jsonl"
echo "[DONE] final predictions: $OUT/final_predictions.jsonl"
