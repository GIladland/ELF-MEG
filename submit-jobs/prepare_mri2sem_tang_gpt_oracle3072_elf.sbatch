#!/usr/bin/env bash
#SBATCH --job-name=prep_tgpt3072
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=submit-jobs/prepare_mri2sem_tang_gpt_oracle3072_elf-%j.out
#SBATCH --error=submit-jobs/prepare_mri2sem_tang_gpt_oracle3072_elf-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
MRI2SEM_RUN=${MRI2SEM_RUN:-/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_tang_openai_gpt_lanczos_seg10_mean_lags4_stride2_v1}
DATASET_NPZ=${DATASET_NPZ:-$MRI2SEM_RUN/dataset_tang_openai_gpt_l9_ctx5_lanczos_d1-2-3-4_seg10.npz}
METADATA_JSON=${METADATA_JSON:-$MRI2SEM_RUN/dataset_tang_openai_gpt_l9_ctx5_lanczos_d1-2-3-4_seg10.json}
PREDICTED_TEST_NPZ=${PREDICTED_TEST_NPZ:-/data/engs-pnpl/glandau/elf-cache/mri2sem/uts01_lanczos_seg10_test_pred3072.npz}
OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-cache/tang_gpt/oracle3072}
PREFIX=${PREFIX:-uts01_tang_gpt_lanczos_seg10_oracle3072}

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

python scripts/export_mri2sem_oracle_to_elf_npz.py \
  --dataset-npz "$DATASET_NPZ" \
  --metadata-json "$METADATA_JSON" \
  --predicted-test-npz "$PREDICTED_TEST_NPZ" \
  --output-dir "$OUTPUT_DIR" \
  --prefix "$PREFIX" \
  --layout both \
  --expected-dim 3072 \
  --delay-dim 768 \
  --expected-train 11725 \
  --expected-val 266 \
  --expected-test 107
