#!/usr/bin/env bash
set -euo pipefail

ROOT_DEFAULT=/disk5/cvlab/from252/custom-whisper-dev
PROJECT_ROOT=${PROJECT_ROOT:-$ROOT_DEFAULT}
if [[ ! -d "$PROJECT_ROOT" ]]; then
  PROJECT_ROOT=$(pwd)
fi
cd "$PROJECT_ROOT"

PY=${PY:-/disk5/cvlab/from252/envs/custom-whisper-mm/bin/python}
if [[ ! -x "$PY" ]]; then
  PY=${PYTHON:-python}
fi
export PYTHONPATH="$PROJECT_ROOT/scripts:$PROJECT_ROOT:${PYTHONPATH:-}"

B_OUT_DEFAULT=outputs/flickr8k_b1_b2_union_normalized_20260708
VAL_UNION_NBEST=""
TEST_UNION_NBEST=""
VAL_CURRENT_PREDICTIONS=""
TEST_CURRENT_PREDICTIONS=""
B2_PREDICTIONS=""
OUTPUT_ROOT=outputs/flickr8k_c2_replacement_decision_$(date +%Y%m%d_%H%M%S)
MAX_CANDIDATES=50
GRID_MODE=quick
TRAIN_EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_c2_replacement_decision.sh \
    --val-union-nbest PATH \
    --test-union-nbest PATH \
    --val-current-predictions PATH \
    --test-current-predictions PATH \
    --output-root PATH \
    [--b2-predictions PATH] [--max-candidates-per-sample 50] [--quick-grid|--full-grid]

If union nbest paths are omitted, the script tries B2 defaults under:
  outputs/flickr8k_b1_b2_union_normalized_20260708/

Current-best prediction paths are required unless one of the built-in candidate
paths exists on the server.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --val-union-nbest) VAL_UNION_NBEST="$2"; shift 2 ;;
    --test-union-nbest) TEST_UNION_NBEST="$2"; shift 2 ;;
    --val-current-predictions) VAL_CURRENT_PREDICTIONS="$2"; shift 2 ;;
    --test-current-predictions) TEST_CURRENT_PREDICTIONS="$2"; shift 2 ;;
    --b2-predictions) B2_PREDICTIONS="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --max-candidates-per-sample) MAX_CANDIDATES="$2"; shift 2 ;;
    --quick-grid) GRID_MODE=quick; shift ;;
    --full-grid) GRID_MODE=full; shift ;;
    --max-grid-evals) TRAIN_EXTRA_ARGS+=(--max-grid-evals "$2"); shift 2 ;;
    --selection-method) TRAIN_EXTRA_ARGS+=(--selection-method "$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

first_existing() {
  for path in "$@"; do
    if [[ -n "$path" && -s "$path" ]]; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

B_OUT=$B_OUT_DEFAULT
VAL_UNION_NBEST=${VAL_UNION_NBEST:-$(first_existing \
  "$B_OUT/normalized_nbest/val_full_union_top80_norm.jsonl" \
  "$B_OUT/rescore_union/val_full_union_top80_A0_A1_A2_A5.jsonl" \
  "$B_OUT/union_nbest/val_full_union_top80.jsonl" || true)}
TEST_UNION_NBEST=${TEST_UNION_NBEST:-$(first_existing \
  "$B_OUT/normalized_nbest/test_full_union_top80_norm.jsonl" \
  "$B_OUT/rescore_union/test_full_union_top80_A0_A1_A2_A5.jsonl" \
  "$B_OUT/union_nbest/test_full_union_top80.jsonl" || true)}
B2_PREDICTIONS=${B2_PREDICTIONS:-$(first_existing \
  "$B_OUT/B2_full_union_norm_grid/predictions_best.jsonl" \
  "$B_OUT/B2_full_union_norm_grid/predictions.jsonl" || true)}

VAL_CURRENT_PREDICTIONS=${VAL_CURRENT_PREDICTIONS:-$(first_existing \
  outputs/flickr8k_val_tuned_postprocess_20260705/current_best_val_predictions.jsonl \
  outputs/flickr8k_val_tuned_postprocess_20260705/val_predictions_best.jsonl \
  outputs/flickr8k_val_tuned_postprocess_20260705/post_rescore_beam20_val_tuned/val_predictions.jsonl || true)}
TEST_CURRENT_PREDICTIONS=${TEST_CURRENT_PREDICTIONS:-$(first_existing \
  outputs/flickr8k_val_tuned_postprocess_20260705/current_best_test_predictions.jsonl \
  outputs/flickr8k_val_tuned_postprocess_20260705/test_predictions_best.jsonl \
  outputs/flickr8k_val_tuned_postprocess_20260705/post_rescore_beam20_val_tuned/test_predictions.jsonl || true)}

require_file() {
  local label="$1"
  local path="$2"
  if [[ -z "$path" || ! -s "$path" ]]; then
    echo "[ERROR] missing $label: $path" >&2
    exit 1
  fi
}

require_file "--val-union-nbest" "$VAL_UNION_NBEST"
require_file "--test-union-nbest" "$TEST_UNION_NBEST"
require_file "--val-current-predictions" "$VAL_CURRENT_PREDICTIONS"
require_file "--test-current-predictions" "$TEST_CURRENT_PREDICTIONS"

mkdir -p "$OUTPUT_ROOT"/{pairs,features,train,eval,compare}

echo "[C2] output_root=$OUTPUT_ROOT"
echo "[C2] val_union=$VAL_UNION_NBEST"
echo "[C2] test_union=$TEST_UNION_NBEST"
echo "[C2] val_current=$VAL_CURRENT_PREDICTIONS"
echo "[C2] test_current=$TEST_CURRENT_PREDICTIONS"
echo "[C2] b2_predictions=${B2_PREDICTIONS:-<missing>}"
echo "[C2] grid=$GRID_MODE max_candidates=$MAX_CANDIDATES"

"$PY" scripts/build_replacement_decision_data.py \
  --nbest-jsonl "$VAL_UNION_NBEST" \
  --current-predictions "$VAL_CURRENT_PREDICTIONS" \
  --output-jsonl "$OUTPUT_ROOT/pairs/val_replacement_pairs.jsonl" \
  --max-candidates-per-sample "$MAX_CANDIDATES" \
  --prediction-field auto

"$PY" scripts/build_replacement_decision_data.py \
  --nbest-jsonl "$TEST_UNION_NBEST" \
  --current-predictions "$TEST_CURRENT_PREDICTIONS" \
  --output-jsonl "$OUTPUT_ROOT/pairs/test_replacement_pairs.jsonl" \
  --max-candidates-per-sample "$MAX_CANDIDATES" \
  --prediction-field auto

"$PY" scripts/add_replacement_features.py \
  --pairs-jsonl "$OUTPUT_ROOT/pairs/val_replacement_pairs.jsonl" \
  --output-jsonl "$OUTPUT_ROOT/features/val_replacement_pairs_features.jsonl"

"$PY" scripts/add_replacement_features.py \
  --pairs-jsonl "$OUTPUT_ROOT/pairs/test_replacement_pairs.jsonl" \
  --output-jsonl "$OUTPUT_ROOT/features/test_replacement_pairs_features.jsonl"

GRID_ARG=--quick-grid
if [[ "$GRID_MODE" == "full" ]]; then
  GRID_ARG=--full-grid
fi

"$PY" scripts/train_replacement_decider.py \
  --val-pairs-jsonl "$OUTPUT_ROOT/features/val_replacement_pairs_features.jsonl" \
  --output-dir "$OUTPUT_ROOT/train/rule_grid_${GRID_MODE}" \
  --method rule_grid \
  "$GRID_ARG" \
  "${TRAIN_EXTRA_ARGS[@]}"

"$PY" scripts/eval_replacement_decider.py \
  --test-pairs-jsonl "$OUTPUT_ROOT/features/test_replacement_pairs_features.jsonl" \
  --rules-json "$OUTPUT_ROOT/train/rule_grid_${GRID_MODE}/rules.json" \
  --current-predictions "$TEST_CURRENT_PREDICTIONS" \
  --output-dir "$OUTPUT_ROOT/eval/rule_grid_${GRID_MODE}"

if [[ -n "${B2_PREDICTIONS:-}" && -s "$B2_PREDICTIONS" ]]; then
  "$PY" scripts/compare_replacement_vs_rerank.py \
    --current-predictions "$TEST_CURRENT_PREDICTIONS" \
    --rerank-predictions "$B2_PREDICTIONS" \
    --replacement-predictions "$OUTPUT_ROOT/eval/rule_grid_${GRID_MODE}/predictions.jsonl" \
    --nbest-jsonl "$TEST_UNION_NBEST" \
    --output-dir "$OUTPUT_ROOT/compare/rule_grid_${GRID_MODE}"
else
  echo "[C2] skip compare: B2 predictions not found"
fi

echo "[C2_DONE] metrics:"
cat "$OUTPUT_ROOT/eval/rule_grid_${GRID_MODE}/metrics.json"
