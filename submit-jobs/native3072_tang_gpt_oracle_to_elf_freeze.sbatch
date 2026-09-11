#!/usr/bin/env bash
#SBATCH --job-name=tgpt3072_elf
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=submit-jobs/native3072_tang_gpt_oracle_to_elf_freeze-%j.out
#SBATCH --error=submit-jobs/native3072_tang_gpt_oracle_to_elf_freeze-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
export ORACLE_DIR=${ORACLE_DIR:-$DATA_ROOT/tang_gpt/oracle3072}
export NPZ_PATH=${NPZ_PATH:-$ORACLE_DIR/uts01_tang_gpt_lanczos_seg10_oracle3072_train_val.npz}
export SOURCE_NPZ=$NPZ_PATH
export INPUT_KEY=${INPUT_KEY:-input_embeddings}
export SENTENCE_KEY=${SENTENCE_KEY:-sentence}

export RUN_TAG=${RUN_TAG:-native3072_tang_gpt_oracle_seg10_freezeelf_30ep_b16_20260824}
export OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/$RUN_TAG}
export TARGET_LATENTS_CACHE=${TARGET_LATENTS_CACHE:-$ORACLE_DIR/${RUN_TAG}_t5small_latents_fp16.npy}

export INIT_E2E_CHECKPOINT=${INIT_E2E_CHECKPOINT:-/data/engs-pnpl/glandau/elf-runs/tang_gpt_storychunks_decoderprob90_continue_from_current_best_30ep_b32_a100_20260823/best.pt}
export INIT_E2E_ADAPTER_MISMATCH=skip
export FREEZE_ELF=1
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export LAST_N_BLOCKS=-1

export EPOCHS=${EPOCHS:-30}
export BATCH_SIZE=${BATCH_SIZE:-16}
export VAL_NUM_EXAMPLES=266
export EVAL_NUM_EXAMPLES=266
export EVAL_EVERY=${EVAL_EVERY:-1000}
export RETRIEVAL_EVAL_EVERY=${RETRIEVAL_EVAL_EVERY:-2000}
export RETRIEVAL_NUM_EXAMPLES=266
export RETRIEVAL_BATCH_SIZE=${RETRIEVAL_BATCH_SIZE:-256}
export NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-32}
export TRAIN_TIMESTEP_STEPS=${TRAIN_TIMESTEP_STEPS:-32}

export LR=${LR:-1e-4}
export CFG_SCALE=1.0
export SELF_COND_CFG_SCALE=1.0
export COND_DROPOUT_PROB=0.0
export DENOISER_LOSS_WEIGHT=0.1
export DECODER_LOSS_WEIGHT=5.0
export DECODER_PROB=0.9
export DECODER_NOISE_SCALE=1.0
export TRAIN_TARGET_MASK_MODE=full
export SEED=7

export USE_WANDB=${USE_WANDB:-1}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-native3072_tang_gpt_elf_bridge_20260824}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$RUN_TAG}
export WANDB_NOTES=${WANDB_NOTES:-Native 3072-D bridge: oracle mean-pooled 10-TR Tang/OpenAI-GPT layer-9 context-5 vectors, hstack delays 1/2/3/4. Initialize latest Tang-GPT ELF weights, skip the incompatible 768-D adapter, freeze ELF, and fit a fresh 3072-D adapter. Train on 11,725 rows and validate on the held-out 266-row tail; the 107 test rows are excluded.}
export SAVE_FINAL_CHECKPOINT=1

if [[ ! -f "$NPZ_PATH" ]]; then
  echo "Missing oracle train+val NPZ: $NPZ_PATH" >&2
  echo "Run submit-jobs/prepare_mri2sem_tang_gpt_oracle3072_elf.sbatch first." >&2
  exit 2
fi
if [[ ! -f "$INIT_E2E_CHECKPOINT" ]]; then
  echo "Missing Tang-GPT initialization checkpoint: $INIT_E2E_CHECKPOINT" >&2
  exit 2
fi

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
