#!/usr/bin/env bash
#SBATCH --job-name=fmri384_val_swp
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4
#SBATCH --output=submit-jobs/eval_fmri_mapped384_validation_sampling_sweep-%A_%a.out
#SBATCH --error=submit-jobs/eval_fmri_mapped384_validation_sampling_sweep-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}

case "$SLURM_ARRAY_TASK_ID" in
  0) sampling_steps=16; cfg_scale=0.5 ;;
  1) sampling_steps=32; cfg_scale=0.5 ;;
  2) sampling_steps=32; cfg_scale=1.0 ;;
  3) sampling_steps=64; cfg_scale=0.5 ;;
  4) sampling_steps=64; cfg_scale=1.0 ;;
  *) echo "Unsupported task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

export BRIDGE_RUN=minilm384_full_unfreeze_continue_werselect_ep75_b16_20260829
export SOURCE_NPZ=/data/engs-pnpl/glandau/elf-runs/delayed_minilm1536_to_exact_center10_minilm384_linear_train11725_val266_20260829/mapped_mri2sem_val_minilm384.npz
export RUN_TAG=fmri_delayed1536_to_exact384_LINEARVALSELECTED_BRAINVAL_s${sampling_steps}_cfg${cfg_scale}_seed7_n266_20260829
export EVAL_NUM_EXAMPLES=266
export RETRIEVAL_NUM_EXAMPLES=266
export NUM_SAMPLING_STEPS=$sampling_steps
export CFG_SCALE=$cfg_scale
export SEED=7
export WANDB_RUN_GROUP=fmri_delayed1536_mapped_exact384_brainval_sampling_20260829
export WANDB_NOTES="Validation-only sampler sweep for the leakage-free fMRI delayed-MiniLM1536 -> validation-selected linear exact-MiniLM384 -> frozen ELF route. No test labels are used for sampler selection."

exec bash "$PROJECT_ROOT/submit-jobs/eval_full_minilm384_clockfix_test.sbatch"
