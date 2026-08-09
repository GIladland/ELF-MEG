#!/usr/bin/env bash
#SBATCH --job-name=sva_m2s_e2e
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --output=submit-jobs/sherlock_sva_meg2sem_minilm_e2e_finetune_dt8z7o55-%j.out
#SBATCH --error=submit-jobs/sherlock_sva_meg2sem_minilm_e2e_finetune_dt8z7o55-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
RUN_TAG=${RUN_TAG:-sherlock_sva_meg2sem_minilm_e2e_finetune_dt8z7o55}

export PROJECT_ROOT
export RUN_TAG
export OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/$RUN_TAG}

export EVAL_ONLY=${EVAL_ONLY:-0}
export TRAIN_MEG2SEM=${TRAIN_MEG2SEM:-1}
export TRAIN_SEMANTIC_PROJECTOR=${TRAIN_SEMANTIC_PROJECTOR:-1}
export UNFREEZE_ELF=${UNFREEZE_ELF:-1}
export UNFREEZE_LAST_N_BLOCKS=${UNFREEZE_LAST_N_BLOCKS:--1}

export BATCH_SIZE=${BATCH_SIZE:-4}
export EPOCHS=${EPOCHS:-20}
export LR=${LR:-5e-6}
export MEG2SEM_LR=${MEG2SEM_LR:-1e-5}
export SEMANTIC_PROJECTOR_LR=${SEMANTIC_PROJECTOR_LR:-5e-6}
export ELF_LR=${ELF_LR:-2e-6}
export EVAL_EVERY=${EVAL_EVERY:-50}
export RETRIEVAL_EVAL_EVERY=${RETRIEVAL_EVAL_EVERY:-50}
export INTERFACE_MONITOR_EVERY=${INTERFACE_MONITOR_EVERY:-5}
export INTERFACE_MONITOR_BATCH_SIZE=${INTERFACE_MONITOR_BATCH_SIZE:-32}
export SAVE_EVAL_CHECKPOINTS=${SAVE_EVAL_CHECKPOINTS:-1}
export EVAL_CHECKPOINT_TOP_K=${EVAL_CHECKPOINT_TOP_K:-3}

export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-meg2sem_minilm_e2e_elf}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$RUN_TAG}

exec bash "$PROJECT_ROOT/submit-jobs/sherlock_sva_meg2sem_minilm_direct_elf_dt8z7o55.sbatch"
