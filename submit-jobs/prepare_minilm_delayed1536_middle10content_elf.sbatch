#!/usr/bin/env bash
#SBATCH --job-name=prep_mini1536_m10
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=submit-jobs/prepare_minilm_delayed1536_middle10content_elf-%j.out
#SBATCH --error=submit-jobs/prepare_minilm_delayed1536_middle10content_elf-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
MRI2SEM_ROOT=${MRI2SEM_ROOT:-/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM}
MRI2SEM_RUN=${MRI2SEM_RUN:-$MRI2SEM_ROOT/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1}
SEGMENT_NPZ=${SEGMENT_NPZ:-$MRI2SEM_RUN/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz}
METADATA_JSON=${METADATA_JSON:-$MRI2SEM_RUN/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.json}
PREDICTIONS_NPZ=${PREDICTIONS_NPZ:-$MRI2SEM_RUN/predictions_test.npz}
STIMULUS_DATA_ROOT=${STIMULUS_DATA_ROOT:-/data/engs-pnpl/glandau/semantic-decoding}

BASE_DIR=${BASE_DIR:-/data/engs-pnpl/glandau/elf-cache/minilm/oracle1536_delayed_seg10}
BASE_PREFIX=${BASE_PREFIX:-uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_oracle1536}
PREDICTED_FILENAME=${PREDICTED_FILENAME:-uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_pred1536_test.npz}

OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-cache/minilm/oracle1536_delayed_seg10_middle10content_clockfix}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_oracle1536_middle10content_clockfix}
PREDICTED_OUTPUT=${PREDICTED_OUTPUT:-$OUTPUT_DIR/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_pred1536_test_middle10content_clockfix.npz}

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
mkdir -p "$BASE_DIR" "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

python scripts/export_mri2sem_minilm_delayed_to_elf.py \
  --dataset-npz "$SEGMENT_NPZ" \
  --metadata-json "$METADATA_JSON" \
  --predictions-npz "$PREDICTIONS_NPZ" \
  --output-dir "$BASE_DIR" \
  --prefix "$BASE_PREFIX" \
  --predicted-filename "$PREDICTED_FILENAME" \
  --expected-dim 1536 \
  --delay-dim 384 \
  --expected-train 11725 \
  --expected-val 266 \
  --expected-test 107

python scripts/relabel_mri2sem_oracle_elf_middle_words.py \
  --input-dir "$BASE_DIR" \
  --input-prefix "$BASE_PREFIX" \
  --output-dir "$OUTPUT_DIR" \
  --output-prefix "$OUTPUT_PREFIX" \
  --metadata-json "$METADATA_JSON" \
  --stimulus-data-root "$STIMULUS_DATA_ROOT" \
  --mri2sem-src "$MRI2SEM_ROOT/src" \
  --predicted-test-npz "$BASE_DIR/$PREDICTED_FILENAME" \
  --predicted-test-output "$PREDICTED_OUTPUT" \
  --word-count 10 \
  --maximize-content-words \
  --min-content-words 2 \
  --content-search-radius-sec 10 \
  --expected-dim 1536 \
  --expected-train 11725 \
  --expected-val 266 \
  --expected-test 107

echo "prepared=$OUTPUT_DIR prefix=$OUTPUT_PREFIX predicted=$PREDICTED_OUTPUT"
