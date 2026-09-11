#!/usr/bin/env bash
#SBATCH --job-name=cal_mri_minilm
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --array=0-1
#SBATCH --output=submit-jobs/calibrate_predicted_minilm_mapper-%A_%a.out
#SBATCH --error=submit-jobs/calibrate_predicted_minilm_mapper-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}

case "$SLURM_ARRAY_TASK_ID" in
  0) mapper_variant=mlp ;;
  1) mapper_variant=linear ;;
  *) exit 2 ;;
esac

mapper_dir=/data/engs-pnpl/glandau/elf-runs/tang3072_to_minilm384_center10_clockfix_${mapper_variant}_20260828
prediction_dir=$DATA_ROOT/mri2sem/uts01_lanczos_seg10_split_predictions
target_dir=$DATA_ROOT/tang_gpt/oracle3072_middle10content_clockfix_minilm
prefix=uts01_tang_gpt_lanczos_seg10_oracle3072_middle10content_clockfix
output_dir=/data/engs-pnpl/glandau/elf-runs/tang3072_to_minilm384_center10_clockfix_${mapper_variant}_valpred_calibrated_20260828

"$TMP_ENV_ROOT/bin/python" "$PROJECT_ROOT/scripts/calibrate_predicted_minilm_mapper.py" \
  --mapper-checkpoint "$mapper_dir/best.pt" \
  --predicted-val-npz "$prediction_dir/predictions_val.npz" \
  --predicted-test-npz "$prediction_dir/predictions_test.npz" \
  --target-train-val-npz "$target_dir/${prefix}_train_val_minilm.npz" \
  --target-test-npz "$target_dir/${prefix}_test_minilm.npz" \
  --output-dir "$output_dir" \
  --folds 5 \
  --device cpu
