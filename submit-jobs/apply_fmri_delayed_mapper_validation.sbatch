#!/usr/bin/env bash
#SBATCH --job-name=map_fmri_val384
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=submit-jobs/apply_fmri_delayed_mapper_validation-%j.out
#SBATCH --error=submit-jobs/apply_fmri_delayed_mapper_validation-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}

linear_run=/data/engs-pnpl/glandau/elf-runs/delayed_minilm1536_to_exact_center10_minilm384_linear_train11725_val266_20260829

"$TMP_ENV_ROOT/bin/python" "$PROJECT_ROOT/scripts/apply_semantic_mapper_to_predictions.py" \
  --mapper-checkpoint "$linear_run/best.pt" \
  --predictions-npz "$DATA_ROOT/minilm/mri2sem_delayed1536_split_predictions_20260829/predictions_val.npz" \
  --target-template-npz "$DATA_ROOT/tang_gpt/oracle3072_middle10content_clockfix_minilm/uts01_tang_gpt_lanczos_seg10_oracle3072_middle10content_clockfix_train_val_minilm.npz" \
  --target-split val \
  --output "$linear_run/mapped_mri2sem_val_minilm384.npz" \
  --device cuda
