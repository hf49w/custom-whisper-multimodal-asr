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

Run untrained ASR/CLIP/MBR/length grid reranking. Grid parameters must be
selected on validation or development n-best files; the test file is evaluated
once with the selected parameters.

```bash
python scripts/rerank_nbest_candidates.py \
  --input-jsonl /path/a9/test_beam20_nbest.jsonl \
  --tune-jsonl /path/a9/val_beam20_nbest.jsonl \
  --output-dir /path/a9/mbr_grid_val_tuned_test_beam20 \
  --clip-model-name /path/clip-vit-base-patch32 \
  --selection-method bootstrap_stable \
  --bootstrap-iters 500 --bootstrap-std-weight 0.5 \
  --device cuda --no-download
```

For train-subset + validation joint selection, first sample a fixed train
subset manifest, dump its n-best candidates with the same checkpoint/beam
settings, then concatenate dev n-best files for parameter selection:

```bash
python scripts/build_rerank_dev_manifest.py \
  --train-manifest /path/train_manifest.jsonl \
  --output-manifest /path/train_subset_4000_seed42.jsonl \
  --sample-size 4000 --seed 42

python scripts/rerank_nbest_candidates.py \
  --input-jsonl /path/a9/test_beam20_nbest.jsonl \
  --dev-nbest /path/a9/val_beam20_nbest.jsonl /path/a9/train_subset_beam20_nbest.jsonl \
  --output-dir /path/a9/mbr_grid_dev_tuned_test_beam20 \
  --selection-method bootstrap_stable \
  --bootstrap-iters 500 --bootstrap-ratio 1.0 --bootstrap-std-weight 0.5 \
  --clip-model-name /path/clip-vit-base-patch32 \
  --device cuda --no-download
```

When teacher-forcing rescoring features are present in the candidate JSONL,
use the constrained teacher grid to avoid unconstrained test-specific weights:

```bash
python scripts/rerank_nbest_candidates.py \
  --input-jsonl /path/a9/test_beam20_rescored.jsonl \
  --dev-nbest /path/a9/val_beam20_rescored.jsonl /path/a9/train_subset_beam20_rescored.jsonl \
  --output-dir /path/a9/post_rescore_constrained_dev_tuned \
  --selection-method bootstrap_stable \
  --constrained-teacher-grid \
  --teacher-a0 0.05 --teacher-a5 0.05 \
  --teacher-w-grid 0,0.1,0.2,0.3,0.4,0.5 \
  --device cuda --no-download
```

`rerank_nbest_candidates.py` writes `best_by_raw_val`,
`best_by_bootstrap_stable`, `grid_top20_raw_val.jsonl`,
`grid_top20_bootstrap_stable.jsonl`, and per-parameter bootstrap
`mean/std/min/max` fields into `grid_metrics.jsonl`.

If a test grid has already been run for diagnostic analysis, inspect
generalization without using test for official selection:

```bash
python scripts/analyze_grid_generalization.py \
  --val-grid /path/a9/val_grid_metrics.jsonl \
  --test-grid /path/a9/test_grid_metrics.jsonl \
  --output-dir /path/a9/grid_generalization_analysis
```

Current official result: the best accepted A9/A10-style result is the
validation-tuned post-rescore beam20 run, with test WER `2.0713%` and CER
`0.8338%`. The lower `2.0391%` post-rescore beam20 number is a test-tuned
analysis result only; it demonstrates search potential but must not be reported
as the formal test score.

## A12/A13 image-semantic candidate reranking

A12/A13 do not retrain Whisper/ASR and do not pass image captions or tags into
the Whisper prompt. Image semantics are extracted offline and used only as
candidate reranking features. All official numbers must select parameters on
validation/development candidates and evaluate test exactly once.

Extract or normalize image semantics:

```bash
python scripts/extract_image_semantics.py \
  --manifest /path/val_manifest.jsonl \
  --output-jsonl /path/image_semantics/val_semantics.jsonl \
  --model-type blip2 \
  --model-name /local/blip2-or-blip-caption-model \
  --device cuda --no-download

# If captions/tags were produced elsewhere:
python scripts/extract_image_semantics.py \
  --manifest /path/val_manifest.jsonl \
  --existing-semantics-jsonl /path/existing_captions_tags.jsonl \
  --model-type none \
  --output-jsonl /path/image_semantics/val_semantics.jsonl
```

Add semantic features to n-best candidates. Run the same transform for val and
test; use shuffled/disabled versions only as controls.

```bash
python scripts/add_image_semantic_features_to_nbest.py \
  --input-jsonl /path/a9/val_beam20_rescored.jsonl \
  --image-semantics-jsonl /path/image_semantics/val_semantics.jsonl \
  --output-jsonl /path/a12/val_beam20_sem_true.jsonl \
  --semantics-mode true

python scripts/add_image_semantic_features_to_nbest.py \
  --input-jsonl /path/a9/test_beam20_rescored.jsonl \
  --image-semantics-jsonl /path/image_semantics/test_semantics.jsonl \
  --output-jsonl /path/a12/test_beam20_sem_true.jsonl \
  --semantics-mode true
```

Run A12 semantic grid reranking. The semantic score adds
`e1*caption_sim + e2*tag_overlap + e3*visual_gain` through the generic
extra-weight mechanism. To prove that the gain uses image semantics, repeat the
same command on shuffled and disabled semantic-feature JSONL files.

```bash
python scripts/rerank_nbest_candidates.py \
  --input-jsonl /path/a12/test_beam20_sem_true.jsonl \
  --tune-jsonl /path/a12/val_beam20_sem_true.jsonl \
  --output-dir /path/a12/true_semantic_grid \
  --selection-method bootstrap_stable \
  --semantic-feature-grid \
  --constrained-teacher-grid \
  --teacher-a0 0.05 --teacher-a5 0.05 \
  --teacher-w-grid 0,0.1,0.2,0.3,0.4,0.5 \
  --caption-sim-values 0,0.02,0.05,0.1,0.2 \
  --tag-overlap-values 0,0.02,0.05,0.1,0.2 \
  --visual-gain-values 0,0.05,0.1,0.2,0.5
```

Run A13 conservative semantic gated reranking. A13 keeps top-1 unless the
selected candidate passes ASR-drop, MBR-gain, visual-gain, rank, and
supported noun/verb changed-word gates.

```bash
python scripts/semantic_conservative_rerank.py \
  --input-jsonl /path/a12/test_beam20_sem_true.jsonl \
  --tune-jsonl /path/a12/val_beam20_sem_true.jsonl \
  --output-dir /path/a13/true_semantic_conservative \
  --constrained-teacher-grid \
  --teacher-a0 0.05 --teacher-a5 0.05 --teacher-w 0.1 \
  --caption-sim-weight 0.05 \
  --tag-overlap-weight 0.05 \
  --visual-gain-weight 0.1
```

Optional A14 keyword logit bias decodes directly from the A1 checkpoint and
adds small positive logits bias to image keyword tokens. Select `beta` on val,
then evaluate the selected value once on test. Always compare true keywords
against shuffled and disabled keywords.

```bash
python scripts/decode_with_image_keyword_bias.py \
  --checkpoint-path /path/A1/checkpoints/best_val_loss.pt \
  --manifest-path /path/val_manifest.jsonl \
  --image-semantics-jsonl /path/image_semantics/val_semantics.jsonl \
  --output-dir /path/a14/val_true_keywords \
  --keyword-mode true \
  --beta-values 0.02,0.05,0.1 \
  --beam-size 20 --device cuda --no-download
```

Status: A12/A13 image-semantic reranking did not improve the official result.
The true/shuffled/disabled semantic ablations selected zero semantic weights and
produced matching outcomes, so captions/tags did not provide reliable
validation-generalizing signal. Do not report A12/A13 as using image semantics
unless true semantics clearly outperform shuffled and disabled controls.

## B1/B2 union candidates and normalized post-rescore

B1/B2 do not train ASR models. They try to improve candidate selection by
building a larger candidate union across decoding sources and using only
per-sample normalized/delta features. Official parameters must be selected on
validation/development data; test is evaluated once.

Build a union candidate pool. Each `SOURCE=path` can be A0 beam20, A1
beam10/20/30, A2 beam20, A5 beam20, or any compatible n-best JSONL.

```bash
python scripts/build_union_nbest.py \
  --input A0=/path/a0_beam20.jsonl \
  --input A1b10=/path/a1_beam10.jsonl \
  --input A1b20=/path/a1_beam20.jsonl \
  --input A1b30=/path/a1_beam30.jsonl \
  --input A2=/path/a2_beam20.jsonl \
  --input A5=/path/a5_beam20.jsonl \
  --primary-source A1b20 \
  --top-k 80 \
  --output-jsonl /path/b1/val_union_top80.jsonl
```

Add normalized B2 features:

```bash
python scripts/add_normalized_rerank_features.py \
  --input-jsonl /path/b1/val_union_top80.jsonl \
  --output-jsonl /path/b2/val_union_top80_norm.jsonl
```

If the union candidates need teacher-forcing scores from several large-v3
models, rescore them with sequential model loading. This is the default behavior
of `rescore_nbest_with_models.py` and avoids OOM on 24GB GPUs by loading one
model, writing the updated JSONL, releasing it, then loading the next model.

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/rescore_nbest_with_models.py \
  --input-jsonl /path/b1/val_union_top80.jsonl \
  --output-jsonl /path/b1/val_union_top80_A0_A1_A2_A5.jsonl \
  --whisper-model-spec A0=/path/large-v3.pt \
  --checkpoint A1=/path/A1_best_val_loss.pt \
  --checkpoint A2=/path/A2_best_val_loss.pt \
  --checkpoint A5=/path/A5_best_val_loss.pt \
  --visual-model-name /path/clip-vit-base-patch32 \
  --device cuda --no-download --candidate-batch-size 8 --resume-output
```

Run normalized post-rescore with validation-selected weights:

```bash
python scripts/rerank_nbest_candidates.py \
  --input-jsonl /path/b2/test_union_top80_norm.jsonl \
  --tune-jsonl /path/b2/val_union_top80_norm.jsonl \
  --output-dir /path/b2/norm_grid_test_once \
  --selection-method bootstrap_stable \
  --bootstrap-iters 500 --bootstrap-ratio 1.0 --bootstrap-std-weight 0.5 \
  --normalized-grid \
  --w-mbr-values 0,0.1,0.2,0.5,1.0 \
  --w-a1-values 0,0.1,0.2,0.5,1.0 \
  --w-a2-values 0,0.05,0.1,0.2,0.5 \
  --w-a0-values 0,0.05,0.1,0.2 \
  --w-a5-values 0,0.05,0.1,0.2 \
  --w-len-values=-0.1,-0.05,0,0.05
```

Analyze what the n-best oracle can fix but the current reranker misses:

```bash
python scripts/analyze_error_types.py \
  --nbest-jsonl /path/b2/test_union_top80_norm.jsonl \
  --predictions-jsonl /path/current_best_predictions.jsonl \
  --output-dir /path/b2/error_types
```

## C2 replacement-decision reranker

C2 does not train or modify the ASR model. It changes the reranking problem from
"score every candidate and pick the highest" to a conservative replacement
decision:

1. keep the current official best prediction by default;
2. consider candidate replacements from the full-union n-best pool;
3. select validation-tuned gates and weights that decide whether one candidate
   is safe enough to replace the current prediction;
4. apply the selected rules once on test.

This is intended for the case where the oracle is strong but global candidate
scoring generalizes poorly. The important diagnostics are:

- `improved_samples`: current-best was wrong and C2 selected a better candidate;
- `worsened_samples`: C2 selected a candidate with more word errors;
- `current_correct_but_replaced_count`: highest-risk failure mode;
- `missed_oracle_cases`: a better candidate existed but C2 did not select it.

Run the default quick grid:

```bash
bash scripts/run_c2_replacement_decision.sh \
  --val-union-nbest /path/b2/val_full_union_top80_norm.jsonl \
  --test-union-nbest /path/b2/test_full_union_top80_norm.jsonl \
  --val-current-predictions /path/current_best_val_predictions.jsonl \
  --test-current-predictions /path/current_best_test_predictions.jsonl \
  --b2-predictions /path/b2/predictions_best.jsonl \
  --output-root outputs/replacement_outputs/c2_quick \
  --quick-grid
```

Run the full grid only when enough CPU time is available:

```bash
bash scripts/run_c2_replacement_decision.sh \
  --val-union-nbest /path/b2/val_full_union_top80_norm.jsonl \
  --test-union-nbest /path/b2/test_full_union_top80_norm.jsonl \
  --val-current-predictions /path/current_best_val_predictions.jsonl \
  --test-current-predictions /path/current_best_test_predictions.jsonl \
  --output-root outputs/replacement_outputs/c2_full \
  --full-grid
```

For a bounded full-grid sweep, add for example `--max-grid-evals 200000`.

Inspect final metrics:

```bash
cat outputs/replacement_outputs/c2_quick/eval/rule_grid_quick/metrics.json
cat outputs/replacement_outputs/c2_quick/train/rule_grid_quick/rules.json
cat outputs/replacement_outputs/c2_quick/compare/rule_grid_quick/metrics.json
```

Test data must not be used for grid selection. Only report test numbers from
`eval_replacement_decider.py` using rules selected on validation data.

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
