#!/usr/bin/env bash
set -euo pipefail

# Required environment variables are paths on the training server. No GPU id is
# embedded; set DEVICE or CUDA_VISIBLE_DEVICES externally.
: "${TRAIN_MANIFEST:?Set TRAIN_MANIFEST}"
: "${VAL_MANIFEST:?Set VAL_MANIFEST}"
: "${TEST_MANIFEST:?Set TEST_MANIFEST}"
: "${WHISPER_MODEL:?Set WHISPER_MODEL to a local Whisper checkpoint}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

EXPERIMENT="${1:-all}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CLIP_MODEL="${CLIP_MODEL:-}"
BLIP2_MODEL="${BLIP2_MODEL:-}"
A6_INIT_CHECKPOINT="${A6_INIT_CHECKPOINT:-${OUTPUT_ROOT}/A4_clipseq_decoder_prompt_k16_shuffle_rank/checkpoints/best_val_loss.pt}"
A7_CHECKPOINT="${A7_CHECKPOINT:-${A6_INIT_CHECKPOINT}}"
RERANK_CLIP_MODEL="${RERANK_CLIP_MODEL:-${CLIP_MODEL}}"

train_common=(
  python scripts/train_visspeech_custom_whisper_fuser.py
  --train-manifest "${TRAIN_MANIFEST}"
  --val-manifest "${VAL_MANIFEST}"
  --whisper-model "${WHISPER_MODEL}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --freeze-whisper
  --freeze-visual-encoder
  --no-download
)

require_clip() {
  if [[ -z "${CLIP_MODEL}" ]]; then
    echo "CLIP_MODEL must point to a local CLIP/EVA-compatible checkpoint" >&2
    exit 2
  fi
}

run_a0() {
  python scripts/eval_whisper_baseline.py \
    --whisper-model "${WHISPER_MODEL}" \
    --manifest-path "${TEST_MANIFEST}" \
    --output-root "${OUTPUT_ROOT}/A0_whisper_select_speech" \
    --device "${DEVICE}" --beam-size 5
}

run_a1() {
  "${train_common[@]}" \
    --output-root "${OUTPUT_ROOT}/A1_blank_decoder_prefix_k16" \
    --visual-encoder none --visual-fuser select_speech \
    --fusion-location decoder_prefix --decoder-prompt-adapter blank_prefix \
    --decoder-prompt-len 16 --lr 1e-3
}

run_clip_prompt() {
  local name="$1" length="$2"
  require_clip
  "${train_common[@]}" \
    --output-root "${OUTPUT_ROOT}/${name}" \
    --visual-encoder clip --clip-model-name "${CLIP_MODEL}" --clip-return-sequence \
    --visual-fuser select_speech --fusion-location decoder_prefix \
    --decoder-prompt-adapter resampler --decoder-prompt-len "${length}" --lr 1e-3
}

run_a4() {
  require_clip
  "${train_common[@]}" \
    --output-root "${OUTPUT_ROOT}/A4_clipseq_decoder_prompt_k16_shuffle_rank" \
    --visual-encoder clip --clip-model-name "${CLIP_MODEL}" --clip-return-sequence \
    --visual-fuser select_speech --fusion-location decoder_prefix \
    --decoder-prompt-adapter resampler --decoder-prompt-len 16 \
    --loss-rank-shuffle --loss-rank-weight 0.1 --loss-rank-margin 0.2 --lr 1e-3
}

run_a5() {
  require_clip
  if [[ -z "${BLIP2_MODEL}" ]]; then
    echo "A5 requires BLIP2_MODEL=/local/blip2/checkpoint (no automatic download)" >&2
    exit 2
  fi
  "${train_common[@]}" \
    --output-root "${OUTPUT_ROOT}/A5_blip2_qformer_decoder_prompt" \
    --visual-encoder clip --clip-model-name "${CLIP_MODEL}" --clip-return-sequence \
    --visual-fuser select_speech --fusion-location decoder_prefix \
    --decoder-prompt-adapter blip2_qformer --blip2-model-name "${BLIP2_MODEL}" \
    --decoder-prompt-len 16 --lr 1e-4
}

run_a6() {
  require_clip
  "${train_common[@]}" \
    --output-root "${OUTPUT_ROOT}/A6_decoder_prompt_lora" \
    --init-from "${A6_INIT_CHECKPOINT}" \
    --visual-encoder clip --clip-model-name "${CLIP_MODEL}" --clip-return-sequence \
    --visual-fuser select_speech --fusion-location decoder_prefix \
    --decoder-prompt-adapter resampler --decoder-prompt-len 16 \
    --loss-rank-shuffle --loss-rank-weight 0.1 --loss-rank-margin 0.2 \
    --enable-decoder-lora --lora-rank 4 --lora-alpha 16 \
    --lora-last-n-layers 4 --lr 2e-5
}

run_a7() {
  if [[ -z "${RERANK_CLIP_MODEL}" ]]; then
    echo "A7 requires RERANK_CLIP_MODEL (or CLIP_MODEL) pointing to a local CLIP checkpoint" >&2
    exit 2
  fi
  python scripts/eval_clip_rerank.py \
    --enable-clip-rerank --checkpoint-path "${A7_CHECKPOINT}" \
    --manifest-path "${TEST_MANIFEST}" --output-root "${OUTPUT_ROOT}/A7_clip_rerank" \
    --clip-model-name "${RERANK_CLIP_MODEL}" --visual-model-name "${CLIP_MODEL}" \
    --beam-size 10 --rerank-n-best 5 --clip-rerank-lambda 0.05 0.1 0.2 \
    --device "${DEVICE}" --no-download
}

case "${EXPERIMENT}" in
  A0|a0) run_a0 ;;
  A1|a1) run_a1 ;;
  A2|a2) run_clip_prompt A2_clipseq_decoder_prompt_k16 16 ;;
  A3|a3) run_clip_prompt A3_clipseq_decoder_prompt_k32 32 ;;
  A4|a4) run_a4 ;;
  A5|a5) run_a5 ;;
  A6|a6) run_a6 ;;
  A7|a7) run_a7 ;;
  all)
    run_a0
    run_a1
    run_clip_prompt A2_clipseq_decoder_prompt_k16 16
    run_clip_prompt A3_clipseq_decoder_prompt_k32 32
    run_a4
    run_a5
    run_a6
    run_a7
    ;;
  *) echo "Usage: $0 {A0|A1|A2|A3|A4|A5|A6|A7|all}" >&2; exit 2 ;;
esac
