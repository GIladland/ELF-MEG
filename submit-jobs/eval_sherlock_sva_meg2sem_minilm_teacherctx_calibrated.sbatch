#!/usr/bin/env bash
#SBATCH --job-name=sva_mlm_cal_ev
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=submit-jobs/eval_sherlock_sva_meg2sem_minilm_teacherctx_calibrated-%j.out
#SBATCH --error=submit-jobs/eval_sherlock_sva_meg2sem_minilm_teacherctx_calibrated-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
oracle_run=sherlock_sva_meg2sem_corpus_minilm_oracle_overfit_full_k64_ep50_b64_20260828
oracle_checkpoint=/data/engs-pnpl/glandau/elf-runs/$oracle_run/best.pt
calibration_run=${CALIBRATION_RUN:-sherlock_sva_meg2sem_dt8z7o55_l2_teacherctx_projector_ep10_b32_20260829}

export RUN_TAG=${RUN_TAG:-${calibration_run}_eval512}
export OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/$RUN_TAG}
export ELF_CHECKPOINT=${ELF_CHECKPOINT:-$oracle_checkpoint}
export SEMANTIC_PROJECTOR_CHECKPOINT=${SEMANTIC_PROJECTOR_CHECKPOINT:-$oracle_checkpoint}
export INIT_E2E_CHECKPOINT=${INIT_E2E_CHECKPOINT:-/data/engs-pnpl/glandau/elf-runs/$calibration_run/best_adapter.pt}
export MEG2SEM_CHECKPOINT_SELECTION=${MEG2SEM_CHECKPOINT_SELECTION:-pinned}
export MEG2SEM_OUTPUT_NORMALIZATION=${MEG2SEM_OUTPUT_NORMALIZATION:-always}

export EVAL_ONLY=1
export TRAIN_MEG2SEM=0
export TRAIN_SEMANTIC_PROJECTOR=0
export UNFREEZE_ELF=0
export TEACHER_CONTEXT_LOSS_WEIGHT=0.0
export VAL_NUM_EXAMPLES=512
export EVAL_NUM_EXAMPLES=256
export RETRIEVAL_NUM_EXAMPLES=512
export RETRIEVAL_BATCH_SIZE=256

export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-sherlock_sva_meg2sem_minilm_teacherctx_calibrated_eval}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$RUN_TAG}

exec bash "$PROJECT_ROOT/submit-jobs/sherlock_sva_meg2sem_minilm_direct_elf_dt8z7o55.sbatch"
