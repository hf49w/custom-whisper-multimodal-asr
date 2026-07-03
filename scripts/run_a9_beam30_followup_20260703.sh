#!/usr/bin/env bash
set -euo pipefail

cd /disk5/cvlab/from252/custom-whisper-dev
export PATH=/disk5/cvlab/from252/envs/custom-whisper-mm/bin:$PATH
export CONDA_PREFIX=/disk5/cvlab/from252/envs/custom-whisper-mm
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

PY=/disk5/cvlab/from252/envs/custom-whisper-mm/bin/python
A1_CKPT=outputs/flickr8k_decoder_prompt_large_v3_A1_rerun_20260626_165442/A1_blank_decoder_prefix_k16/checkpoints/best_val_loss.pt
TEST_MANIFEST=data/flickr8k/prepared/splits/by_image_id_seed42_val10_test10/test_manifest.jsonl
CLIP_MODEL=/disk5/cvlab/from252/custom-whisper/data/models/clip/clip-vit-base-patch32
OUT=outputs/flickr8k_a9_candidate_rerank_20260703
LOGDIR=logs/a9_candidate_rerank_20260703

mkdir -p "$OUT" "$LOGDIR"

log() {
  echo "[$(date '+%F %T')] $*"
}

line_count() {
  local file="$1"
  if [ -s "$file" ]; then
    wc -l < "$file" | tr -d ' '
  else
    echo 0
  fi
}

wait_complete_jsonl() {
  local file="$1"
  local expected="$2"
  while true; do
    lines=$(line_count "$file")
    if [ "$lines" -ge "$expected" ]; then
      log "ready file=$file lines=$lines"
      return 0
    fi
    log "waiting file=$file lines=$lines expected=$expected"
    sleep 300
  done
}

wait_free_gpu() {
  while true; do
    for gpu in 3 0 1; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')
      if [ "${used:-999999}" -le 1024 ]; then
        echo "$gpu"
        return 0
      fi
    done
    log "waiting for free GPU among 3/0/1"
    sleep 120
  done
}

wait_complete_jsonl "$OUT/val_beam30_nbest.jsonl" 4000

if [ "$(line_count "$OUT/test_beam30_nbest.jsonl")" -lt 4000 ]; then
  gpu=$(wait_free_gpu)
  log "START dump test beam30 gpu=$gpu" | tee -a "$LOGDIR/dump_test_beam30.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/dump_nbest_candidates.py \
    --checkpoint-path "$A1_CKPT" \
    --manifest-path "$TEST_MANIFEST" \
    --beam-size 30 \
    --n-best 30 \
    --output-jsonl "$OUT/test_beam30_nbest.jsonl" \
    --device cuda \
    --no-download \
    --log-every 20 >>"$LOGDIR/dump_test_beam30.log" 2>&1
  log "DONE dump test beam30 gpu=$gpu" | tee -a "$LOGDIR/dump_test_beam30.log"
fi

gpu=$(wait_free_gpu)
log "START beam30 MBR/CLIP grid gpu=$gpu" | tee -a "$LOGDIR/beam30_rerank.log"
CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/rerank_nbest_candidates.py \
  --input-jsonl "$OUT/test_beam30_nbest.jsonl" \
  --output-dir "$OUT/test_beam30_mbr_clip_grid" \
  --clip-model-name "$CLIP_MODEL" \
  --device cuda \
  --no-download \
  --a-values "0,0.5,1" \
  --b-values "0,0.01,0.02,0.05,0.1,0.2" \
  --c-values "0,0.05,0.1,0.2,0.5,1" \
  --d-values "0,0.01,0.02,0.05,0.1" \
  --log-every 100 >>"$LOGDIR/beam30_rerank.log" 2>&1
log "DONE beam30 MBR/CLIP grid" | tee -a "$LOGDIR/beam30_rerank.log"

gpu=$(wait_free_gpu)
log "START beam30 train ridge reranker gpu=$gpu" | tee -a "$LOGDIR/beam30_rerank.log"
CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/train_candidate_reranker.py \
  --val-jsonl "$OUT/val_beam30_nbest.jsonl" \
  --output-pkl "$OUT/reranker_beam30_ridge.pkl" \
  --output-dir "$OUT/train_reranker_beam30_ridge" \
  --clip-model-name "$CLIP_MODEL" \
  --device cuda \
  --no-download \
  --model-type ridge \
  --ridge-alpha-values "0.001,0.01,0.1,1,10,100" \
  --cv-folds 5 \
  --log-every 100 >>"$LOGDIR/beam30_rerank.log" 2>&1
log "DONE beam30 train ridge reranker" | tee -a "$LOGDIR/beam30_rerank.log"

gpu=$(wait_free_gpu)
log "START beam30 eval ridge reranker gpu=$gpu" | tee -a "$LOGDIR/beam30_rerank.log"
CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/eval_candidate_reranker.py \
  --test-jsonl "$OUT/test_beam30_nbest.jsonl" \
  --reranker-pkl "$OUT/reranker_beam30_ridge.pkl" \
  --output-dir "$OUT/eval_reranker_beam30_ridge" \
  --clip-model-name "$CLIP_MODEL" \
  --device cuda \
  --no-download \
  --log-every 100 >>"$LOGDIR/beam30_rerank.log" 2>&1
log "DONE beam30 eval ridge reranker" | tee -a "$LOGDIR/beam30_rerank.log"

log "beam30 follow-up finished"
