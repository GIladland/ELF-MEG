#!/usr/bin/env bash
#SBATCH --job-name=map_minilm_bridge
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36G
#SBATCH --output=submit-jobs/train_mapped_minilm384_adapter_bridge_clockfix-%j.out
#SBATCH --error=submit-jobs/train_mapped_minilm384_adapter_bridge_clockfix-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
mapper_variant=${MAPPER_VARIANT:-mlp}
mapper_dir=${MAPPER_DIR:-/data/engs-pnpl/glandau/elf-runs/tang3072_to_minilm384_center10_clockfix_${mapper_variant}_20260828}

export NPZ_PATH=$mapper_dir/mapped_train_val_minilm384.npz
export SOURCE_NPZ=$NPZ_PATH
export RUN_TAG=center10_clockfix_tangmapped_${mapper_variant}_minilm384_freezeelf_adapter_train11725_val266_ep10_20260828
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$DATA_ROOT/tang_gpt/semantic_bridge_full/${RUN_TAG}_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/tang_gpt_seg10_podcast_storychunks8to18_t12_minilm_elf_fullmask_k64_full_150ep_b32_short12h_a100/best.pt
export INIT_E2E_ADAPTER_MISMATCH=error

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
export SEMANTIC_INPUT_PROJECTION_DIM=0
export FREEZE_ELF=1
export LR=1e-4
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_NOISE_SCALE=2.5
export TRAIN_TARGET_MASK_MODE=full
export USE_WANDB=1
export WANDB_RUN_GROUP=center10_clockfix_tangmapped_minilm384_bridge_20260828
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Phase 1 mapped semantic bridge: freeze ELF and calibrate the pretrained MiniLM-384 semantic adapter on 11,725 clock-corrected Tang-3072→MiniLM mapper outputs, with a 266-row held-out story validation tail."
export SAVE_FINAL_CHECKPOINT=1

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
