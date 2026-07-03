#!/usr/bin/env bash
set -u

cd /disk5/cvlab/from252/custom-whisper-dev || exit 1
export PATH=/disk5/cvlab/from252/envs/custom-whisper-mm/bin:$PATH
export CONDA_PREFIX=/disk5/cvlab/from252/envs/custom-whisper-mm
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

PY=/disk5/cvlab/from252/envs/custom-whisper-mm/bin/python
WHISPER_MODEL=/disk5/cvlab/from252/custom-whisper/data/models/whisper/large-v3.pt
CLIP_MODEL=/disk5/cvlab/from252/custom-whisper/data/models/clip/clip-vit-base-patch32
TRAIN_MANIFEST=data/flickr8k/prepared/splits/by_image_id_seed42_val10_test10/train_manifest.jsonl
VAL_MANIFEST=data/flickr8k/prepared/splits/by_image_id_seed42_val10_test10/val_manifest.jsonl
TEST_MANIFEST=data/flickr8k/prepared/splits/by_image_id_seed42_val10_test10/test_manifest.jsonl
A1_DOMAIN_CKPT=outputs/flickr8k_decoder_prompt_large_v3_A1_rerun_20260626_165442/A1_blank_decoder_prefix_k16/checkpoints/best_val_loss.pt

OUT=outputs/flickr8k_decoder_prompt_large_v3/A8_a1_domain_delta_clip_k16
EVAL_OUT=outputs/flickr8k_decode_only_20260702/A8_a1_domain_delta_clip_k16_beam5
LOGDIR=logs/domain_delta_20260702
TRAIN_LOG="$LOGDIR/A8_a1_domain_delta_clip_k16.train.log"
EVAL_LOG="$LOGDIR/A8_a1_domain_delta_clip_k16.eval.log"
TARGET_EPOCHS=10

mkdir -p "$LOGDIR" "$OUT" "$EVAL_OUT"

log() {
  echo "[$(date '+%F %T')] $*"
}

wait_free_gpu() {
  while true; do
    for gpu in 0 1 5; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')
      if [ "${used:-999999}" -le 1024 ]; then
        echo "$gpu"
        return 0
      fi
    done
    log "waiting for one of GPUs 0/1/5 to be free" >&2
    sleep 120
  done
}

completed_epochs() {
  "$PY" - "$OUT/train_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = Path(sys.argv[1])
if not summary.is_file():
    print(0)
    raise SystemExit(0)
try:
    data = json.loads(summary.read_text(encoding="utf-8"))
except Exception:
    print(0)
    raise SystemExit(0)
print(int(data.get("completed_epochs") or 0))
PY
}

train_done() {
  epochs=$(completed_epochs)
  [ "${epochs:-0}" -ge "$TARGET_EPOCHS" ] && [ -s "$OUT/checkpoints/best_val_loss.pt" ]
}

log "initial delay so prompt-context wait queue can claim the next free GPU first"
sleep 300

while ! train_done; do
  GPU=$(wait_free_gpu)
  export CUDA_VISIBLE_DEVICES="$GPU"
  log "START A8 train on physical GPU $GPU" | tee -a "$TRAIN_LOG"

  RESUME_ARGS=()
  if [ -s "$OUT/checkpoints/last.pt" ]; then
    RESUME_ARGS=(--resume-from "$OUT/checkpoints/last.pt" --allow-relocated-paths)
    log "resume_from=$OUT/checkpoints/last.pt" | tee -a "$TRAIN_LOG"
  fi

  "$PY" scripts/train_visspeech_custom_whisper_fuser.py \
    --train-manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --whisper-model "$WHISPER_MODEL" \
    --epochs "$TARGET_EPOCHS" \
    --batch-size 8 \
    --device cuda \
    --freeze-whisper \
    --freeze-visual-encoder \
    --no-download \
    --output-root "$OUT" \
    --visual-encoder clip \
    --clip-model-name "$CLIP_MODEL" \
    --clip-return-sequence \
    --visual-fuser select_speech \
    --fusion-location decoder_prefix \
    --decoder-prompt-adapter domain_delta_resampler \
    --decoder-prompt-len 16 \
    --domain-prefix-from "$A1_DOMAIN_CKPT" \
    --domain-delta-scale 1.0 \
    --loss-rank-shuffle \
    --loss-rank-weight 0.1 \
    --loss-rank-margin 0.2 \
    --lr 1e-3 \
    "${RESUME_ARGS[@]}" >>"$TRAIN_LOG" 2>&1
  rc=$?

  if train_done; then
    log "DONE A8 train target reached" | tee -a "$TRAIN_LOG"
    break
  fi

  log "A8 train exited rc=$rc before target; retry after cooldown" | tee -a "$TRAIN_LOG"
  sleep 600
done

GPU=$(wait_free_gpu)
export CUDA_VISIBLE_DEVICES="$GPU"
log "START A8 decode-only eval on physical GPU $GPU" | tee -a "$EVAL_LOG"
"$PY" scripts/eval_decode_only_oracle_rerank.py \
  --checkpoint-path "$OUT/checkpoints/best_val_loss.pt" \
  --manifest-path "$TEST_MANIFEST" \
  --output-root "$EVAL_OUT" \
  --device cuda \
  --beam-size 5 \
  --n-best 5 \
  --skip-if-exists \
  --log-every 20 \
  --clip-model-name "$CLIP_MODEL" \
  --clip-rerank-lambda 0.02 0.05 0.1 0.2 >>"$EVAL_LOG" 2>&1
rc=$?
log "DONE A8 decode-only eval rc=$rc" | tee -a "$EVAL_LOG"
exit "$rc"
