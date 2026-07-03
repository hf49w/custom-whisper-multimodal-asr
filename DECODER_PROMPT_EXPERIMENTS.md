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
  `--decoder-prompt-dropout`, `--decoder-prompt-insert`,
  `--decoder-prompt-special-tokens`, and `--decoder-prompt-missing`.
  For `after_special_tokens`, the default `-1` resolves to the exact length of
  `sot_sequence_including_notimestamps` (or `sot_sequence`). During decoding the
  insertion point is computed as `sot_index + len(sot_sequence)`, so an
  `sot_prev` prompt does not shift the image prefix into user text.
- Freezing: `--freeze-whisper`, `--freeze-visual-encoder` and their `--no-*`
  counterparts.
- Ranking: `--loss-rank-shuffle`, `--loss-rank-weight`,
  `--loss-rank-margin`.
- Token weighting: `--visual-token-weighting none|pos` and
  `--visual-token-weight`. POS mode requires every manifest row to contain a
  `visual_pos_mask` JSON/array aligned with the encoded label positions. Missing
  masks raise an error. Unit-weight fallback is available only when explicitly
  requested with `--allow-missing-visual-pos-mask`.
- LoRA: `--enable-decoder-lora`, `--lora-rank`, `--lora-alpha`,
  `--lora-dropout`, `--lora-last-n-layers`, and `--lora-targets`.
- Local-only models: `--no-download`, `--clip-model-name`, and
  `--blip2-model-name`.

Diagnostics are available in `eval_visspeech_custom_whisper_fuser.py` via
`--shuffle-images-at-eval`, `--zero-prefix-at-eval`,
`--use-trained-blank-prefix-at-eval`, and `--disable-image-at-eval`. Zero prefix
uses all-zero embeddings. Trained blank prefix is accepted only for an A1-style
checkpoint whose adapter is `blank_prefix`. The old hidden
`--blank-prefix-at-eval` alias retains its former zero-prefix behavior.

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

## A9 candidate reranking

A9 does not change or retrain the ASR model. It reuses an existing checkpoint
and only changes how n-best decode candidates are selected. The intended use is
with the A1 blank decoder-prefix checkpoint and the same decode-only path as
`eval_decode_only_oracle_rerank.py`.

Dump validation and test candidates:

```bash
python scripts/dump_nbest_candidates.py \
  --checkpoint-path /path/A1/checkpoints/best_val_loss.pt \
  --manifest-path /path/val_manifest.jsonl \
  --beam-size 20 --n-best 20 \
  --output-jsonl /path/a9/val_beam20_nbest.jsonl \
  --device cuda --no-download

python scripts/dump_nbest_candidates.py \
  --checkpoint-path /path/A1/checkpoints/best_val_loss.pt \
  --manifest-path /path/test_manifest.jsonl \
  --beam-size 20 --n-best 20 \
  --output-jsonl /path/a9/test_beam20_nbest.jsonl \
  --device cuda --no-download
```

Run untrained ASR/CLIP/MBR/length grid reranking:

```bash
python scripts/rerank_nbest_candidates.py \
  --input-jsonl /path/a9/test_beam20_nbest.jsonl \
  --output-dir /path/a9/mbr_grid_test_beam20 \
  --clip-model-name /path/clip-vit-base-patch32 \
  --device cuda --no-download
```

Train a validation-set reranker:

```bash
python scripts/train_candidate_reranker.py \
  --val-jsonl /path/a9/val_beam20_nbest.jsonl \
  --output-pkl /path/a9/reranker_beam20.pkl \
  --output-dir /path/a9/train_reranker_beam20 \
  --clip-model-name /path/clip-vit-base-patch32 \
  --device cuda --no-download
```

Evaluate the trained reranker on test candidates:

```bash
python scripts/eval_candidate_reranker.py \
  --test-jsonl /path/a9/test_beam20_nbest.jsonl \
  --reranker-pkl /path/a9/reranker_beam20.pkl \
  --output-dir /path/a9/eval_reranker_beam20 \
  --device cuda --no-download
```

The A9 reports include top-1 WER/CER, top-k oracle curves for
`k=1,5,10,20,30,50`, MBR/grid metrics, validation cross-validation metrics for
the trained reranker, selected predictions, error cases, and samples where the
oracle could fix top-1 but the reranker did not select that candidate.
