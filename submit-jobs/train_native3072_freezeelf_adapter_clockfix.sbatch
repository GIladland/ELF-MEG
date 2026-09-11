#!/usr/bin/env bash
#SBATCH --job-name=native_adapter
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36G
#SBATCH --output=submit-jobs/train_native3072_freezeelf_adapter_clockfix-%j.out
#SBATCH --error=submit-jobs/train_native3072_freezeelf_adapter_clockfix-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
source_dir=$DATA_ROOT/tang_gpt/oracle3072_middle10content_clockfix
prefix=uts01_tang_gpt_lanczos_seg10_oracle3072_middle10content_clockfix

export NPZ_PATH=$source_dir/${prefix}_train_val.npz
export SOURCE_NPZ=$NPZ_PATH
export RUN_TAG=middle10_clockfix_native3072_meaninit_freezeelf_adapter_train11725_val266_ep10_b16_20260828
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
# The target text and tokenization are identical to the full-unfreeze run, so this cache is reusable.
export TARGET_LATENTS_CACHE=$DATA_ROOT/tang_gpt/semantic_bridge_full/middle10_clockfix_native3072_meaninit_full_unfreeze_train11725_val266_ep30_b16_20260828_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/tang_gpt_storychunks_decoderprob90_continue_from_current_best_30ep_b32_a100_20260823/best.pt
export INIT_E2E_ADAPTER_MISMATCH=mean-blocks

export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export TORCH_PIP_SPEC=${TORCH_PIP_SPEC:-torch==2.7.1}
export TORCH_PIP_INDEX_URL=${TORCH_PIP_INDEX_URL:-https://download.pytorch.org/whl/cu118}

export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=266
export EVAL_NUM_EXAMPLES=266
export RETRIEVAL_NUM_EXAMPLES=266
export RETRIEVAL_EVAL_EVERY=4000
export RETRIEVAL_BATCH_SIZE=256
export EVAL_EVERY=1000
export EPOCHS=10
export BATCH_SIZE=16
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_INPUT_PROJECTION_DIM=768
export FREEZE_ELF=1
export LR=1e-4
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_NOISE_SCALE=2.5
export TRAIN_TARGET_MASK_MODE=full
export USE_WANDB=1
export WANDB_RUN_GROUP=middle10_clockfix_native3072_staged_bridge_20260828
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Stage 1 native bridge: keep ELF fixed while calibrating the pretrained Tang adapter plus learnable mean-initialized 3072-to-768 four-delay projection on 11,725 clock-corrected rows; validate on 266 held-out chunks."
export SAVE_FINAL_CHECKPOINT=1

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
