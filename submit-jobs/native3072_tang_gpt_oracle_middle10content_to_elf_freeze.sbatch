#!/usr/bin/env bash
#SBATCH --job-name=tgpt3072_m10c
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=submit-jobs/native3072_tang_gpt_oracle_middle10content_to_elf_freeze-%j.out
#SBATCH --error=submit-jobs/native3072_tang_gpt_oracle_middle10content_to_elf_freeze-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
export ORACLE_DIR=${ORACLE_DIR:-$DATA_ROOT/tang_gpt/oracle3072_middle10content}
export NPZ_PATH=${NPZ_PATH:-$ORACLE_DIR/uts01_tang_gpt_lanczos_seg10_oracle3072_middle10content_train_val.npz}
export RUN_TAG=${RUN_TAG:-native3072_tang_gpt_oracle_seg10_middle10content_freezeelf_30ep_b32_20260825}
export OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/$RUN_TAG}
export TARGET_LATENTS_CACHE=${TARGET_LATENTS_CACHE:-$ORACLE_DIR/${RUN_TAG}_t5small_latents_fp16.npy}

export TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export TORCH_PIP_SPEC=${TORCH_PIP_SPEC:-torch==2.7.1}
export TORCH_PIP_INDEX_URL=${TORCH_PIP_INDEX_URL:-https://download.pytorch.org/whl/cu118}

export EPOCHS=${EPOCHS:-30}
export BATCH_SIZE=${BATCH_SIZE:-32}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-native3072_tang_gpt_elf_bridge_middle10content_20260825}
export WANDB_NOTES=${WANDB_NOTES:-Native 3072-D oracle Tang-GPT bridge with exactly 10 contiguous TextGrid words per target. The selected window midpoint remains inside the lag-adjusted 20-second segment, maximizes lexical content, and contains at least two audited content words; boundary spill caused by pauses is recorded. ELF is frozen and a fresh 3072-D adapter is trained.}

exec bash "$PROJECT_ROOT/submit-jobs/native3072_tang_gpt_oracle_to_elf_freeze.sbatch"
