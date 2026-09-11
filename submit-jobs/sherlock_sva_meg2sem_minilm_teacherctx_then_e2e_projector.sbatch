#!/usr/bin/env bash
#SBATCH --job-name=sva_mlm_e2eprj
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=submit-jobs/sherlock_sva_meg2sem_minilm_teacherctx_then_e2e_projector-%j.out
#SBATCH --error=submit-jobs/sherlock_sva_meg2sem_minilm_teacherctx_then_e2e_projector-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
oracle_run=sherlock_sva_meg2sem_corpus_minilm_oracle_overfit_full_k64_ep50_b64_20260828
oracle_checkpoint=/data/engs-pnpl/glandau/elf-runs/$oracle_run/best.pt
teacher_run=${TEACHER_RUN:-sherlock_sva_meg2sem_dt8z7o55_l2_teacherctx_projector_ep10_b32_20260829}

export RUN_TAG=${RUN_TAG:-sherlock_sva_meg2sem_dt8z7o55_l2_teacherctx_then_e2e_projector_ep10_b16_20260829}
export OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/$RUN_TAG}
export ELF_CHECKPOINT=${ELF_CHECKPOINT:-$oracle_checkpoint}
export SEMANTIC_PROJECTOR_CHECKPOINT=${SEMANTIC_PROJECTOR_CHECKPOINT:-$oracle_checkpoint}
export INIT_E2E_CHECKPOINT=${INIT_E2E_CHECKPOINT:-/data/engs-pnpl/glandau/elf-runs/$teacher_run/best_adapter.pt}
export MEG2SEM_CHECKPOINT_SELECTION=${MEG2SEM_CHECKPOINT_SELECTION:-pinned}
export MEG2SEM_OUTPUT_NORMALIZATION=${MEG2SEM_OUTPUT_NORMALIZATION:-always}

export EVAL_ONLY=0
export TRAIN_MEG2SEM=0
export TRAIN_SEMANTIC_PROJECTOR=1
export UNFREEZE_ELF=0
export TEACHER_CONTEXT_LOSS_WEIGHT=0.0
export EPOCHS=10
export BATCH_SIZE=16
export LR=1e-5
export SEMANTIC_PROJECTOR_LR=1e-5
export EVAL_EVERY=500
export RETRIEVAL_EVAL_EVERY=1000
export VAL_NUM_EXAMPLES=512
export EVAL_NUM_EXAMPLES=128
export RETRIEVAL_NUM_EXAMPLES=128
export RETRIEVAL_BATCH_SIZE=256
export INTERFACE_MONITOR_EVERY=50
export INTERFACE_MONITOR_BATCH_SIZE=32
export SAVE_EVAL_CHECKPOINTS=1
export EVAL_CHECKPOINT_TOP_K=3

export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-sherlock_sva_meg2sem_minilm_e2e_projector}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$RUN_TAG}

exec bash "$PROJECT_ROOT/submit-jobs/sherlock_sva_meg2sem_minilm_direct_elf_dt8z7o55.sbatch"
