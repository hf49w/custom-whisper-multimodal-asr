# Decoder image soft-prompt experiments

The legacy `encoder_memory`/feature-fuser implementation remains the default. The
new route is enabled with `--fusion-location decoder_prefix`; it prepends image
soft prompts only to decoder self-attention and keeps prediction/loss shapes at
the original token length.

## Smoke tests

Run on a machine with PyTorch and torchvision installed:

```bash
python -m unittest tests.test_decoder_prefix -v
```

The tests use tiny random models and mock visual features. They do not load or
download Whisper, CLIP, BLIP-2, datasets, or checkpoints.

## A0-A7 launcher

All paths are supplied through environment variables; the launcher contains no
absolute paths or GPU ids.

```bash
export TRAIN_MANIFEST=/path/flickr8k_train.jsonl
export VAL_MANIFEST=/path/flickr8k_val.jsonl
export TEST_MANIFEST=/path/flickr8k_test.jsonl
export WHISPER_MODEL=/local/medium.en.pt
export CLIP_MODEL=/local/clip-model
export OUTPUT_ROOT=/experiment/output
export DEVICE=cuda

bash scripts/run_flickr8k_decoder_prompt_experiments.sh A2
```

Use `A0` through `A7`, or `all`. A5 additionally requires
`BLIP2_MODEL=/local/blip2-checkpoint`. A6 defaults to A4's best validation
checkpoint and can be overridden with `A6_INIT_CHECKPOINT`. A7 can be pointed at
A4 or A6 with `A7_CHECKPOINT` and requires `RERANK_CLIP_MODEL` (defaults to
`CLIP_MODEL`). A6 is not a strict frozen-Whisper comparison because decoder LoRA
updates are enabled.

The training script is single-process. To use several GPUs concurrently, launch
independent experiment IDs with different externally assigned `DEVICE` or
`CUDA_VISIBLE_DEVICES` values. It does not hardcode GPU allocation.

## Main flags

- Prefix: `--fusion-location`, `--decoder-prompt-adapter`,
  `--decoder-prompt-len`, `--decoder-prompt-heads`,
  `--decoder-prompt-dropout`, `--decoder-prompt-insert`, and
  `--decoder-prompt-missing`.
- Freezing: `--freeze-whisper`, `--freeze-visual-encoder` and their `--no-*`
  counterparts.
- Ranking: `--loss-rank-shuffle`, `--loss-rank-weight`,
  `--loss-rank-margin`.
- Token weighting: `--visual-token-weighting none|pos` and
  `--visual-token-weight`. POS mode uses `visual_pos_mask` when supplied by a
  dataset; otherwise it intentionally falls back to unit weights.
- LoRA: `--enable-decoder-lora`, `--lora-rank`, `--lora-alpha`,
  `--lora-dropout`, `--lora-last-n-layers`, and `--lora-targets`.
- Local-only models: `--no-download`, `--clip-model-name`, and
  `--blip2-model-name`.

Diagnostics are available in `eval_visspeech_custom_whisper_fuser.py` via
`--shuffle-images-at-eval`, `--blank-prefix-at-eval`, and
`--disable-image-at-eval`.

## CLIP reranking

`scripts/eval_clip_rerank.py` decodes a single padded 30-second segment per
manifest row, retains beam candidates, and reports WER/CER before and after each
lambda. This is suitable for Flickr8K clips but does not implement long-form
segmented n-best merging.

```bash
python scripts/eval_clip_rerank.py \
  --enable-clip-rerank \
  --checkpoint-path /path/best_val_loss.pt \
  --manifest-path /path/flickr8k_test.jsonl \
  --output-root /path/A7 \
  --clip-model-name /local/clip-model \
  --beam-size 10 --rerank-n-best 5 \
  --clip-rerank-lambda 0.05 0.1 0.2 --no-download
```

BLIP-2 Q-Former support is optional and local-only. It loads a local
`Blip2ForConditionalGeneration` checkpoint, retains its Q-Former, and adapts
precomputed visual sequences to Whisper prompts. If `transformers` or the local
checkpoint is absent, model construction fails with an actionable error. No
large model is downloaded automatically when `--no-download` is used.
