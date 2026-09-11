#!/usr/bin/env bash
#SBATCH --job-name=fmri_retcal
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=submit-jobs/eval_fmri_delayed1536_storycv_retrieval_calibration-%j.out
#SBATCH --error=submit-jobs/eval_fmri_delayed1536_storycv_retrieval_calibration-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
PYTHON=${PYTHON:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118/bin/python}
OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/fmri_minilm1536_storycv_retrieval_calibration_test107_20260830}

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

exec "$PYTHON" scripts/evaluate_delayed_retrieval_calibration.py \
  --validation-npz "$DATA_ROOT/minilm/mri2sem_delayed1536_split_predictions_20260829/predictions_val.npz" \
  --test-npz "$DATA_ROOT/minilm/oracle1536_delayed_seg10_middle10content_clockfix/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_pred1536_test_middle10content_clockfix.npz" \
  --output-json "$OUTPUT_DIR/metrics.json" \
  --output-csv "$OUTPUT_DIR/predictions.csv" \
  --delay-count 4
