#!/usr/bin/env bash
#SBATCH --job-name=prep_tgpt3072_m10c
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=submit-jobs/prepare_mri2sem_tang_gpt_oracle3072_middle10content_elf-%j.out
#SBATCH --error=submit-jobs/prepare_mri2sem_tang_gpt_oracle3072_middle10content_elf-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
MRI2SEM_ROOT=${MRI2SEM_ROOT:-/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM}
MRI2SEM_RUN=${MRI2SEM_RUN:-$MRI2SEM_ROOT/runs/uts01_tang_openai_gpt_lanczos_seg10_mean_lags4_stride2_v1}
METADATA_JSON=${METADATA_JSON:-$MRI2SEM_RUN/dataset_tang_openai_gpt_l9_ctx5_lanczos_d1-2-3-4_seg10.json}
STIMULUS_DATA_ROOT=${STIMULUS_DATA_ROOT:-/data/engs-pnpl/glandau/semantic-decoding}
INPUT_DIR=${INPUT_DIR:-/data/engs-pnpl/glandau/elf-cache/tang_gpt/oracle3072}
INPUT_PREFIX=${INPUT_PREFIX:-uts01_tang_gpt_lanczos_seg10_oracle3072}
OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-cache/tang_gpt/oracle3072_middle10content}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-uts01_tang_gpt_lanczos_seg10_oracle3072_middle10content}
PREDICTED_TEST_NPZ=${PREDICTED_TEST_NPZ:-/data/engs-pnpl/glandau/elf-cache/mri2sem/uts01_lanczos_seg10_test_pred3072.npz}
PREDICTED_TEST_OUTPUT=${PREDICTED_TEST_OUTPUT:-$OUTPUT_DIR/uts01_lanczos_seg10_test_pred3072_middle10content.npz}

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

python scripts/relabel_mri2sem_oracle_elf_middle_words.py \
  --input-dir "$INPUT_DIR" \
  --input-prefix "$INPUT_PREFIX" \
  --output-dir "$OUTPUT_DIR" \
  --output-prefix "$OUTPUT_PREFIX" \
  --metadata-json "$METADATA_JSON" \
  --stimulus-data-root "$STIMULUS_DATA_ROOT" \
  --mri2sem-src "$MRI2SEM_ROOT/src" \
  --predicted-test-npz "$PREDICTED_TEST_NPZ" \
  --predicted-test-output "$PREDICTED_TEST_OUTPUT" \
  --word-count 10 \
  --maximize-content-words \
  --min-content-words 2 \
  --content-search-radius-sec 10 \
  --expected-dim 3072 \
  --expected-train 11725 \
  --expected-val 266 \
  --expected-test 107
