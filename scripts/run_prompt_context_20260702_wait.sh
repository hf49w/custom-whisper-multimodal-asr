#!/usr/bin/env bash
set -euo pipefail

cd /disk5/cvlab/from252/custom-whisper-dev
export PATH=/disk5/cvlab/from252/envs/custom-whisper-mm/bin:$PATH
export CONDA_PREFIX=/disk5/cvlab/from252/envs/custom-whisper-mm
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

PY=/disk5/cvlab/from252/envs/custom-whisper-mm/bin/python
MANIFEST=data/flickr8k/prepared/splits/by_image_id_seed42_val10_test10/test_manifest.jsonl
BLIP2=/disk5/cvlab/from252/custom-whisper/data/models/blip2/blip2-opt-2.7b
A1_CKPT=outputs/flickr8k_decoder_prompt_large_v3_A1_rerun_20260626_165442/A1_blank_decoder_prefix_k16/checkpoints/best_val_loss.pt
OUT=outputs/flickr8k_prompt_context_20260702
CONTEXTS="$OUT/blip2_test_image_contexts.jsonl"

mkdir -p logs/prompt_context_20260702 "$OUT"

wait_free_gpu() {
  while true; do
    for gpu in 0 1 5; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
      if [ "${used:-999999}" -le 1024 ]; then
        echo "$gpu"
        return 0
      fi
    done
    echo "[$(date '+%F %T')] waiting for one of GPUs 0/1/5 to be free" >&2
    sleep 120
  done
}

GPU=$(wait_free_gpu)
export CUDA_VISIBLE_DEVICES="$GPU"
echo "[$(date '+%F %T')] using physical GPU $GPU"

echo "[$(date '+%F %T')] START generate_blip2_contexts"
"$PY" scripts/generate_blip2_image_contexts.py \
  --manifest-path "$MANIFEST" \
  --output-jsonl "$CONTEXTS" \
  --model-name "$BLIP2" \
  --device cuda \
  --log-every 20
echo "[$(date '+%F %T')] DONE generate_blip2_contexts"

run_eval() {
  local name="$1"
  local field="$2"
  local shuffle_flag="$3"
  echo "[$(date '+%F %T')] START $name"
  if [ "$shuffle_flag" = "shuffle" ]; then
    "$PY" scripts/eval_decode_only_oracle_rerank.py \
      --checkpoint-path "$A1_CKPT" \
      --manifest-path "$MANIFEST" \
      --output-root "$OUT/$name" \
      --device cuda --beam-size 5 --n-best 5 --skip-if-exists --log-every 20 \
      --prompt-jsonl "$CONTEXTS" --prompt-key image_id --prompt-field "$field" \
      --prompt-mode prompt --prompt-template "{prompt}" --shuffle-prompts
  else
    "$PY" scripts/eval_decode_only_oracle_rerank.py \
      --checkpoint-path "$A1_CKPT" \
      --manifest-path "$MANIFEST" \
      --output-root "$OUT/$name" \
      --device cuda --beam-size 5 --n-best 5 --skip-if-exists --log-every 20 \
      --prompt-jsonl "$CONTEXTS" --prompt-key image_id --prompt-field "$field" \
      --prompt-mode prompt --prompt-template "{prompt}"
  fi
  echo "[$(date '+%F %T')] DONE $name"
}

run_eval A1_20_caption_prompt_true_beam5 caption_prompt true
run_eval A1_20_caption_prompt_shuffle_beam5 caption_prompt shuffle
run_eval A1_20_tags_prompt_true_beam5 tags_prompt true
run_eval A1_20_tags_prompt_shuffle_beam5 tags_prompt shuffle

echo "[$(date '+%F %T')] prompt context queue finished"
