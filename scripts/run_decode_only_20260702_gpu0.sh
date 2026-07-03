#!/usr/bin/env bash
set -euo pipefail

cd /disk5/cvlab/from252/custom-whisper-dev
export PATH=/disk5/cvlab/from252/envs/custom-whisper-mm/bin:$PATH
export CONDA_PREFIX=/disk5/cvlab/from252/envs/custom-whisper-mm
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

MANIFEST=data/flickr8k/prepared/splits/by_image_id_seed42_val10_test10/test_manifest.jsonl
WHISPER=/disk5/cvlab/from252/custom-whisper/data/models/whisper/large-v3.pt
OUT=outputs/flickr8k_decode_only_20260702
PY=/disk5/cvlab/from252/envs/custom-whisper-mm/bin/python

mkdir -p logs/decode_only_20260702 "$OUT"

run_task() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START $name"
  "$@"
  echo "[$(date '+%F %T')] DONE $name"
}

run_task A0_baseline_beam5 \
  "$PY" scripts/eval_decode_only_oracle_rerank.py \
    --whisper-model "$WHISPER" \
    --manifest-path "$MANIFEST" \
    --output-root "$OUT/A0_baseline_beam5" \
    --device cuda --beam-size 5 --n-best 5 --skip-if-exists --log-every 20

run_task A1_20ep_beam5 \
  "$PY" scripts/eval_decode_only_oracle_rerank.py \
    --checkpoint-path outputs/flickr8k_decoder_prompt_large_v3_A1_rerun_20260626_165442/A1_blank_decoder_prefix_k16/checkpoints/best_val_loss.pt \
    --manifest-path "$MANIFEST" \
    --output-root "$OUT/A1_20ep_beam5" \
    --device cuda --beam-size 5 --n-best 5 --skip-if-exists --log-every 20

run_task A2_20ep_beam5 \
  "$PY" scripts/eval_decode_only_oracle_rerank.py \
    --checkpoint-path outputs/flickr8k_decoder_prompt_large_v3/A2_clipseq_decoder_prompt_k16/checkpoints/best_val_loss.pt \
    --manifest-path "$MANIFEST" \
    --output-root "$OUT/A2_20ep_beam5" \
    --device cuda --beam-size 5 --n-best 5 --skip-if-exists --log-every 20

