#!/usr/bin/env bash
set -euo pipefail

cd /disk5/cvlab/from252/custom-whisper-dev
export PATH=/disk5/cvlab/from252/envs/custom-whisper-mm/bin:$PATH
export CONDA_PREFIX=/disk5/cvlab/from252/envs/custom-whisper-mm
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

PY=/disk5/cvlab/from252/envs/custom-whisper-mm/bin/python
A1_CKPT=outputs/flickr8k_decoder_prompt_large_v3_A1_rerun_20260626_165442/A1_blank_decoder_prefix_k16/checkpoints/best_val_loss.pt
VAL_MANIFEST=data/flickr8k/prepared/splits/by_image_id_seed42_val10_test10/val_manifest.jsonl
TEST_MANIFEST=data/flickr8k/prepared/splits/by_image_id_seed42_val10_test10/test_manifest.jsonl
CLIP_MODEL=/disk5/cvlab/from252/custom-whisper/data/models/clip/clip-vit-base-patch32
OUT=outputs/flickr8k_a9_candidate_rerank_20260703
LOGDIR=logs/a9_candidate_rerank_20260703

mkdir -p "$OUT" "$LOGDIR"

log() {
  echo "[$(date '+%F %T')] $*"
}

dump_candidates() {
  local gpu="$1"
  local split="$2"
  local manifest="$3"
  local beam="$4"
  local log_file="$LOGDIR/dump_${split}_beam${beam}.log"
  local output_jsonl="$OUT/${split}_beam${beam}_nbest.jsonl"
  log "START dump split=$split beam=$beam gpu=$gpu output=$output_jsonl" | tee -a "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/dump_nbest_candidates.py \
    --checkpoint-path "$A1_CKPT" \
    --manifest-path "$manifest" \
    --beam-size "$beam" \
    --n-best "$beam" \
    --output-jsonl "$output_jsonl" \
    --device cuda \
    --no-download \
    --skip-if-exists \
    --log-every 20 >>"$log_file" 2>&1
  log "DONE dump split=$split beam=$beam gpu=$gpu" | tee -a "$log_file"
}

run_beam20_rerank() {
  local log_file="$LOGDIR/beam20_rerank.log"
  log "START beam20 MBR/CLIP grid" | tee -a "$log_file"
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/rerank_nbest_candidates.py \
    --input-jsonl "$OUT/test_beam20_nbest.jsonl" \
    --output-dir "$OUT/test_beam20_mbr_clip_grid" \
    --clip-model-name "$CLIP_MODEL" \
    --device cuda \
    --no-download \
    --a-values "0,0.5,1" \
    --b-values "0,0.01,0.02,0.05,0.1,0.2" \
    --c-values "0,0.05,0.1,0.2,0.5,1" \
    --d-values "0,0.01,0.02,0.05,0.1" \
    --log-every 100 >>"$log_file" 2>&1
  log "DONE beam20 MBR/CLIP grid" | tee -a "$log_file"

  log "START beam20 train ridge reranker" | tee -a "$log_file"
  CUDA_VISIBLE_DEVICES=1 "$PY" scripts/train_candidate_reranker.py \
    --val-jsonl "$OUT/val_beam20_nbest.jsonl" \
    --output-pkl "$OUT/reranker_beam20_ridge.pkl" \
    --output-dir "$OUT/train_reranker_beam20_ridge" \
    --clip-model-name "$CLIP_MODEL" \
    --device cuda \
    --no-download \
    --model-type ridge \
    --ridge-alpha-values "0.001,0.01,0.1,1,10,100" \
    --cv-folds 5 \
    --log-every 100 >>"$log_file" 2>&1
  log "DONE beam20 train ridge reranker" | tee -a "$log_file"

  log "START beam20 eval ridge reranker" | tee -a "$log_file"
  CUDA_VISIBLE_DEVICES=1 "$PY" scripts/eval_candidate_reranker.py \
    --test-jsonl "$OUT/test_beam20_nbest.jsonl" \
    --reranker-pkl "$OUT/reranker_beam20_ridge.pkl" \
    --output-dir "$OUT/eval_reranker_beam20_ridge" \
    --clip-model-name "$CLIP_MODEL" \
    --device cuda \
    --no-download \
    --log-every 100 >>"$log_file" 2>&1
  log "DONE beam20 eval ridge reranker" | tee -a "$log_file"
}

dump_candidates 0 val "$VAL_MANIFEST" 20 &
pid_val20=$!
dump_candidates 1 test "$TEST_MANIFEST" 20 &
pid_test20=$!
dump_candidates 3 val "$VAL_MANIFEST" 30 &
pid_val30=$!

wait "$pid_val20"
wait "$pid_test20"
run_beam20_rerank

wait "$pid_val30" || true
log "A9 queue finished"
