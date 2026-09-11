#!/usr/bin/env bash
#SBATCH --job-name=eval_native_train
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=submit-jobs/eval_middle10_native3072_completed_checkpoint_trainhead-%j.out
#SBATCH --error=submit-jobs/eval_middle10_native3072_completed_checkpoint_trainhead-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
export RUN_TAG=middle10_native3072_frozen_completed_trainhead128_eval_20260827
export NPZ_PATH=$DATA_ROOT/tang_gpt/oracle3072_middle10content/uts01_tang_gpt_lanczos_seg10_oracle3072_middle10content_train_val.npz
export SOURCE_NPZ=$NPZ_PATH
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$DATA_ROOT/tang_gpt/semantic_bridge_diagnostics/${RUN_TAG}_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/native3072_tang_gpt_oracle_seg10_middle10content_freezeelf_30ep_b32_20260825/best.pt
export INIT_E2E_ADAPTER_MISMATCH=error

export TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export TORCH_PIP_SPEC=${TORCH_PIP_SPEC:-torch==2.7.1}
export TORCH_PIP_INDEX_URL=${TORCH_PIP_INDEX_URL:-https://download.pytorch.org/whl/cu118}

export NUM_EXAMPLES=128
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=128
export RETRIEVAL_NUM_EXAMPLES=128
export RETRIEVAL_EVAL_EVERY=1
export RETRIEVAL_BATCH_SIZE=256
export EPOCHS=1
export BATCH_SIZE=16
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export FREEZE_ELF=1
export EVAL_ONLY=1
export USE_WANDB=0
export SAVE_FINAL_CHECKPOINT=0
export DENOISER_LOSS_WEIGHT=0.1
export DECODER_LOSS_WEIGHT=5.0
export DECODER_NOISE_SCALE=1.0

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
