#!/usr/bin/env bash
#SBATCH --job-name=pred_mri_val
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=submit-jobs/export_mri2sem_val_predictions-%j.out
#SBATCH --error=submit-jobs/export_mri2sem_val_predictions-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}

run_dir=${RUN_DIR:-/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_tang_openai_gpt_lanczos_seg10_mean_lags4_stride2_v1}
dataset_npz=${DATASET_NPZ:-$run_dir/dataset_tang_openai_gpt_l9_ctx5_lanczos_d1-2-3-4_seg10.npz}
checkpoint=${CHECKPOINT:-$run_dir/model.pt}
output_dir=${OUTPUT_DIR:-$DATA_ROOT/mri2sem/uts01_lanczos_seg10_split_predictions}

"$TMP_ENV_ROOT/bin/python" "$PROJECT_ROOT/scripts/export_mri2sem_split_predictions.py" \
  --mri2sem-root /data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM \
  --dataset-npz "$dataset_npz" \
  --checkpoint "$checkpoint" \
  --output-dir "$output_dir" \
  --splits val test \
  --batch-size 128
