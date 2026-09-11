#!/usr/bin/env bash
#SBATCH --job-name=eval_tgpt3072_m10c
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=submit-jobs/eval_native3072_tang_gpt_oracle_middle10content_vs_mri2sem-%j.out
#SBATCH --error=submit-jobs/eval_native3072_tang_gpt_oracle_middle10content_vs_mri2sem-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
export ORACLE_DIR=${ORACLE_DIR:-$DATA_ROOT/tang_gpt/oracle3072_middle10content}
export RUN_TAG=${RUN_TAG:-native3072_tang_gpt_oracle_seg10_middle10content_freezeelf_30ep_b32_20260825}
export TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/$RUN_TAG}
export ORACLE_TEST_NPZ=${ORACLE_TEST_NPZ:-$ORACLE_DIR/uts01_tang_gpt_lanczos_seg10_oracle3072_middle10content_test.npz}
export PREDICTED_TEST_NPZ=${PREDICTED_TEST_NPZ:-$ORACLE_DIR/uts01_lanczos_seg10_test_pred3072_middle10content.npz}
export EVAL_LATENTS_DIR=${EVAL_LATENTS_DIR:-$ORACLE_DIR}

export TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export TORCH_PIP_SPEC=${TORCH_PIP_SPEC:-torch==2.7.1}
export TORCH_PIP_INDEX_URL=${TORCH_PIP_INDEX_URL:-https://download.pytorch.org/whl/cu118}

exec bash "$PROJECT_ROOT/submit-jobs/eval_native3072_tang_gpt_oracle_vs_mri2sem.sbatch"
