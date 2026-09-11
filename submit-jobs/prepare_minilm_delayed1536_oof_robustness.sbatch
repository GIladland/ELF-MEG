#!/usr/bin/env bash
#SBATCH --job-name=prep_mini_oofmix
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=submit-jobs/prepare_minilm_delayed1536_oof_robustness-%j.out
#SBATCH --error=submit-jobs/prepare_minilm_delayed1536_oof_robustness-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
ORACLE_DIR=${ORACLE_DIR:-/data/engs-pnpl/glandau/elf-cache/minilm/oracle1536_delayed_seg10_middle10content_clockfix}
ORACLE_PREFIX=${ORACLE_PREFIX:-uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_oracle1536_middle10content_clockfix}
FOLD_DIR=${FOLD_DIR:-/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_oof5_30ep_v1}
OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-cache/minilm/oof_robustness}
OUTPUT_NPZ=${OUTPUT_NPZ:-$OUTPUT_DIR/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz}
PYTHON=${PYTHON:-/data/engs-pnpl/glandau/elf-cache/conda-envs/braindiffusion-elf-torch271cu118/bin/python}

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

"$PYTHON" scripts/build_minilm_oof_robustness_npz.py \
  --oracle-train-npz "$ORACLE_DIR/${ORACLE_PREFIX}_train.npz" \
  --oracle-val-npz "$ORACLE_DIR/${ORACLE_PREFIX}_val.npz" \
  --fold-dir "$FOLD_DIR" \
  --output-npz "$OUTPUT_NPZ" \
  --num-folds 5 \
  --expected-train 11725 \
  --expected-val 266 \
  --expected-dim 1536

echo "prepared=$OUTPUT_NPZ"
