#!/usr/bin/env bash
#SBATCH --job-name=mini1536_m10_adapt
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=submit-jobs/minilm_delayed1536_middle10content_freezeelf_adapter-%j.out
#SBATCH --error=submit-jobs/minilm_delayed1536_middle10content_freezeelf_adapter-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
export ORACLE_DIR=${ORACLE_DIR:-$DATA_ROOT/minilm/oracle1536_delayed_seg10_middle10content_clockfix}
export ORACLE_PREFIX=${ORACLE_PREFIX:-uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_oracle1536_middle10content_clockfix}
export NPZ_PATH=${NPZ_PATH:-$ORACLE_DIR/${ORACLE_PREFIX}_train_val.npz}
export SOURCE_NPZ=$NPZ_PATH
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence

export RUN_TAG=${RUN_TAG:-minilm_delayed1536_seg10_middle10content_clockfix_meaninit_freezeelf_adapter_train11725_val266_ep30_b32_20260828}
export OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/$RUN_TAG}
export TARGET_LATENTS_CACHE=${TARGET_LATENTS_CACHE:-$ORACLE_DIR/${RUN_TAG}_t5small_latents_fp16.npy}

# This checkpoint is the matched 384-D MiniLM -> middle-10-word bridge.  The
# new trainable 1536->384 projection starts as the exact mean of the four
# delayed blocks, while the compatible MiniLM adapter and ELF weights load.
export INIT_E2E_CHECKPOINT=${INIT_E2E_CHECKPOINT:-/data/engs-pnpl/glandau/elf-runs/middle10_clockfix_minilm384_full_unfreeze_train11725_val266_ep30_b16_20260828/best.pt}
export INIT_E2E_ADAPTER_MISMATCH=mean-blocks
export SEMANTIC_INPUT_PROJECTION_DIM=384
export SEMANTIC_ADAPTER_KIND=flat
export FREEZE_ELF=1

export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export TORCH_PIP_SPEC=${TORCH_PIP_SPEC:-torch==2.7.1}
export TORCH_PIP_INDEX_URL=${TORCH_PIP_INDEX_URL:-https://download.pytorch.org/whl/cu118}

export NUM_EXAMPLES=${NUM_EXAMPLES:-0}
export VAL_NUM_EXAMPLES=${VAL_NUM_EXAMPLES:-266}
export EVAL_NUM_EXAMPLES=${EVAL_NUM_EXAMPLES:-266}
export RETRIEVAL_NUM_EXAMPLES=${RETRIEVAL_NUM_EXAMPLES:-266}
export RETRIEVAL_EVAL_EVERY=${RETRIEVAL_EVAL_EVERY:-2000}
export RETRIEVAL_BATCH_SIZE=${RETRIEVAL_BATCH_SIZE:-256}
export EVAL_EVERY=${EVAL_EVERY:-1000}
export EPOCHS=${EPOCHS:-30}
export BATCH_SIZE=${BATCH_SIZE:-32}
export CONTEXT_LENGTH=${CONTEXT_LENGTH:-64}
export SEMANTIC_HIDDEN_DIM=${SEMANTIC_HIDDEN_DIM:-4096}
export LAST_N_BLOCKS=${LAST_N_BLOCKS:--1}
export LR=${LR:-1e-4}
export DENOISER_LOSS_WEIGHT=${DENOISER_LOSS_WEIGHT:-1.0}
export DECODER_LOSS_WEIGHT=${DECODER_LOSS_WEIGHT:-1.0}
export DECODER_PROB=${DECODER_PROB:-0.5}
export DECODER_NOISE_SCALE=${DECODER_NOISE_SCALE:-2.5}
export TRAIN_TARGET_MASK_MODE=${TRAIN_TARGET_MASK_MODE:-full}
export NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-32}
export TRAIN_TIMESTEP_STEPS=${TRAIN_TIMESTEP_STEPS:-32}
export USE_WANDB=${USE_WANDB:-1}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-minilm_delayed1536_elf_bridge_middle10content_20260828}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$RUN_TAG}
export WANDB_NOTES=${WANDB_NOTES:-Oracle delayed MiniLM bridge: four ordered 384-D Tang-clock stimulus-delay blocks, mean pooled over 10 TR, paired with an audited contiguous 10-word content-rich TextGrid target. Initialize the matched MiniLM384 middle10 ELF bridge, add a trainable mean-initialized 1536-to-384 projection, freeze ELF, train on 11,725 oracle rows, validate on 266 held-out rows, and exclude all 107 test rows.}
export SAVE_FINAL_CHECKPOINT=1

if [[ ! -f "$NPZ_PATH" ]]; then
  echo "Missing prepared oracle NPZ: $NPZ_PATH" >&2
  exit 2
fi
if [[ ! -f "$INIT_E2E_CHECKPOINT" ]]; then
  echo "Missing MiniLM initialization checkpoint: $INIT_E2E_CHECKPOINT" >&2
  exit 2
fi

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
