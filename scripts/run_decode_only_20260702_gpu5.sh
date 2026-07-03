#!/usr/bin/env bash
set -euo pipefail

cd /disk5/cvlab/from252/custom-whisper-dev
export PATH=/disk5/cvlab/from252/envs/custom-whisper-mm/bin:$PATH
export CONDA_PREFIX=/disk5/cvlab/from252/envs/custom-whisper-mm
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5

MANIFEST=data/flickr8k/prepared/splits/by_image_id_seed42_val10_test10/test_manifest.jsonl
CLIP=/disk5/cvlab/from252/custom-whisper/data/models/clip/clip-vit-base-patch32
OUT=outputs/flickr8k_decode_only_20260702
PY=/disk5/cvlab/from252/envs/custom-whisper-mm/bin/python

mkdir -p logs/decode_only_20260702 "$OUT"

echo "[$(date '+%F %T')] START A1_20ep_beam20_oracle_clip"
"$PY" scripts/eval_decode_only_oracle_rerank.py \
  --checkpoint-path outputs/flickr8k_decoder_prompt_large_v3_A1_rerun_20260626_165442/A1_blank_decoder_prefix_k16/checkpoints/best_val_loss.pt \
  --manifest-path "$MANIFEST" \
  --output-root "$OUT/A1_20ep_beam20_oracle_clip" \
  --device cuda --beam-size 20 --n-best 20 --skip-if-exists --log-every 20 \
  --clip-model-name "$CLIP" \
  --clip-rerank-lambda 0.02 0.05 0.1 0.2
echo "[$(date '+%F %T')] DONE A1_20ep_beam20_oracle_clip"
