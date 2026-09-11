#!/usr/bin/env bash
#SBATCH --job-name=prep_tm_apples
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=submit-jobs/prepare_tang_themoth_apples_meg2sem-%j.out
#SBATCH --error=submit-jobs/prepare_tang_themoth_apples_meg2sem-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
MEG2SEM_ROOT=${MEG2SEM_ROOT:-/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM}
MEG2SEM_PYTHON=${MEG2SEM_PYTHON:-/data/engs-pnpl/glandau/conda/envs/meg2sem/bin/python}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
THEMOTH_ROOT=${THEMOTH_ROOT:-/data/engs-pnpl/datasets/LibriBrain2/TheMoth}
LIBRIBRAIN2_ROOT=${LIBRIBRAIN2_ROOT:-/data/engs-pnpl/datasets/LibriBrain2}
ORACLE_DIR=${ORACLE_DIR:-$DATA_ROOT/minilm/knowntext_unseenbrain_v1}
ORACLE_TRAIN_VAL=${ORACLE_TRAIN_VAL:-$ORACLE_DIR/oracle_exact_minilm384_train_val.npz}
ORACLE_TEST=${ORACLE_TEST:-$ORACLE_DIR/oracle_exact_minilm384_test107.npz}
OUTPUT_ROOT=${OUTPUT_ROOT:-$DATA_ROOT/tang_themoth_apples_exact384_v2}
SIDECAR_ROOT=${SIDECAR_ROOT:-$OUTPUT_ROOT/minilm_sidecars}
PACKED_ROOT=${PACKED_ROOT:-$OUTPUT_ROOT/packed}
SEGMENT_MS=${SEGMENT_MS:-20000}
CENTER_MODE=${CENTER_MODE:-target_center}
PREFIX=${PREFIX:-tang_themoth_exact384_minilm384_seg${SEGMENT_MS}}
TRAIN_CHANNEL_STATS_NPZ=${TRAIN_CHANNEL_STATS_NPZ:-$DATA_ROOT/libribrain_channel_stats/libribrain_channel_stats_cabb9a33e9b9772fa421.npz}

mkdir -p "$SIDECAR_ROOT" "$PACKED_ROOT" "$OUTPUT_ROOT"
cd "$PROJECT_ROOT"

# The two transcript sources differ on occasional filled-pause repeats (for
# example, two versus three consecutive "so" tokens). Exact matching is still
# above 92%, including every locked-test row; the shared clock is independently
# enforced by the sub-100-ms p95 timing audit.
"$MEG2SEM_PYTHON" scripts/export_tang_themoth_minilm_sidecars.py \
  --oracle-train-val "$ORACLE_TRAIN_VAL" \
  --oracle-test "$ORACLE_TEST" \
  --themoth-root "$THEMOTH_ROOT" \
  --output-root "$SIDECAR_ROOT" \
  --summary-json "$OUTPUT_ROOT/alignment_summary.json" \
  --audit-csv "$OUTPUT_ROOT/alignment_rows.csv" \
  --segment-ms "$SEGMENT_MS" \
  --center-mode "$CENTER_MODE" \
  --expected-shared-stories 30 \
  --expected-embedding-dim 384 \
  --expected-word-count 10 \
  --min-exact-text-match-fraction 0.90 \
  --max-clock-error-p95-sec 0.35 \
  --overwrite

train_keys=()
for session in 1 2 3 4 5 7 8 9 10 11 12 13 14 15 16 17 18 20 21 23 25 26 27 28 29 30; do
  train_keys+=(--run-key "0 $session TheMoth 1")
done
val_keys=(
  --run-key "0 6 TheMoth 1"
  --run-key "0 22 TheMoth 1"
  --run-key "0 24 TheMoth 1"
)
test_keys=(--run-key "0 19 TheMoth 1")

common_args=(
  --data-path "$LIBRIBRAIN2_ROOT"
  --semantic-root "$SIDECAR_ROOT"
  --tasks TheMoth
  --books 1
  --set sentences
  --embedding-type MINILM
  --segment-ms "$SEGMENT_MS"
  --progress-every 100
)

export LIBRIBRAIN_CHANNEL_STATS_CACHE=${LIBRIBRAIN_CHANNEL_STATS_CACHE:-$DATA_ROOT/libribrain_channel_stats}

train_output=$PACKED_ROOT/${PREFIX}_train_fp16.npz
val_output=$PACKED_ROOT/${PREFIX}_val_fp16.npz
test_output=$PACKED_ROOT/${PREFIX}_test_fp16.npz

if [[ -f "$train_output" && -f "$TRAIN_CHANNEL_STATS_NPZ" ]]; then
  echo "Reusing completed training pack: $train_output"
  echo "Reusing training-only channel statistics: $TRAIN_CHANNEL_STATS_NPZ"
else
  "$MEG2SEM_PYTHON" "$MEG2SEM_ROOT/utils/pack_sherlock_sva_splits_npz.py" \
    "${common_args[@]}" "${train_keys[@]}" --split train \
    --output "$train_output"
fi

if [[ ! -f "$TRAIN_CHANNEL_STATS_NPZ" ]]; then
  echo "Missing training-only channel statistics: $TRAIN_CHANNEL_STATS_NPZ" >&2
  exit 2
fi

"$MEG2SEM_PYTHON" "$MEG2SEM_ROOT/utils/pack_sherlock_sva_splits_npz.py" \
  "${common_args[@]}" "${val_keys[@]}" --split val \
  --channel-stats-npz "$TRAIN_CHANNEL_STATS_NPZ" \
  --output "$val_output"

"$MEG2SEM_PYTHON" "$MEG2SEM_ROOT/utils/pack_sherlock_sva_splits_npz.py" \
  "${common_args[@]}" "${test_keys[@]}" --split test \
  --channel-stats-npz "$TRAIN_CHANNEL_STATS_NPZ" \
  --output "$test_output"

"$MEG2SEM_PYTHON" "$MEG2SEM_ROOT/utils/check_packed_meg2sem_health.py" \
  --npz "train=$train_output" \
  --npz "val=$val_output" \
  --npz "test=$test_output" \
  --embedding-type MINILM \
  --expect-embedding-dim 384 \
  --expect-segment-ms "$SEGMENT_MS" \
  --max-meg-rows 512 \
  --meg-sample-mode random \
  --out-json "$OUTPUT_ROOT/packed_health.json"

"$MEG2SEM_PYTHON" scripts/validate_tang_themoth_exact384_contract.py \
  --oracle-test-npz "$ORACLE_TEST" \
  --meg-test-npz "$test_output" \
  --story birthofanation \
  --output-json "$OUTPUT_ROOT/exact384_cross_modal_contract.json"

"$MEG2SEM_PYTHON" scripts/compute_semantic_train_mean.py \
  --input-npz "$train_output" \
  --output-npz "$OUTPUT_ROOT/tang_themoth_exact384_minilm384_train_mean.npz" \
  --embedding-key embeddings_minilm

echo "Prepared Tang/TheMoth apples-to-apples MEG2SEM data under $OUTPUT_ROOT"
