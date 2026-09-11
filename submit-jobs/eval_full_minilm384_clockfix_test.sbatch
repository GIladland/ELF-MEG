#!/usr/bin/env bash
#SBATCH --job-name=eval_full_minilm
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --output=submit-jobs/eval_full_minilm384_clockfix_test-%j.out
#SBATCH --error=submit-jobs/eval_full_minilm384_clockfix_test-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
source_dir=$DATA_ROOT/tang_gpt/oracle3072_middle10content_clockfix_minilm
prefix=uts01_tang_gpt_lanczos_seg10_oracle3072_middle10content_clockfix
bridge_run=${BRIDGE_RUN:-middle10_clockfix_minilm384_full_unfreeze_train11725_val266_ep30_b16_20260828}

export NPZ_PATH=${SOURCE_NPZ:-$source_dir/${prefix}_test_minilm.npz}
export RUN_TAG=${RUN_TAG:-eval_${bridge_run}_exact_minilm384}
export SOURCE_NPZ=$NPZ_PATH
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$DATA_ROOT/tang_gpt/semantic_bridge_diagnostics/${RUN_TAG}_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/$bridge_run/best.pt
export INIT_E2E_ADAPTER_MISMATCH=error

export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export TORCH_PIP_SPEC=${TORCH_PIP_SPEC:-torch==2.7.1}
export TORCH_PIP_INDEX_URL=${TORCH_PIP_INDEX_URL:-https://download.pytorch.org/whl/cu118}

export NUM_EXAMPLES=${NUM_EXAMPLES:-0}
export VAL_NUM_EXAMPLES=${VAL_NUM_EXAMPLES:-0}
export EVAL_NUM_EXAMPLES=${EVAL_NUM_EXAMPLES:-107}
export RETRIEVAL_NUM_EXAMPLES=${RETRIEVAL_NUM_EXAMPLES:-$EVAL_NUM_EXAMPLES}
export RETRIEVAL_EVAL_EVERY=${RETRIEVAL_EVAL_EVERY:-1}
export RETRIEVAL_BATCH_SIZE=${RETRIEVAL_BATCH_SIZE:-256}
export EPOCHS=${EPOCHS:-1}
export BATCH_SIZE=${BATCH_SIZE:-16}
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_INPUT_PROJECTION_DIM=0
export FREEZE_ELF=1
export EVAL_ONLY=1
export USE_WANDB=1
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-middle10_clockfix_full_minilm384_test_20260828}
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES=${WANDB_NOTES:-"Held-out exact-MiniLM semantic upper bound for the full clock-corrected center-10-word ELF bridge."}
export SAVE_FINAL_CHECKPOINT=0
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_NOISE_SCALE=2.5

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
