#!/usr/bin/env bash
#SBATCH --job-name=sva_mlm_pair
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --array=0-1
#SBATCH --output=submit-jobs/eval_sherlock_sva_meg2sem_corpus_minilm_overfit-%A_%a.out
#SBATCH --error=submit-jobs/eval_sherlock_sva_meg2sem_corpus_minilm_overfit-%A_%a.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TRAIN_RUN_TAG=${TRAIN_RUN_TAG:-sherlock_sva_meg2sem_corpus_minilm_oracle_overfit_full_k64_ep50_b64_20260828}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/$TRAIN_RUN_TAG}
TRAIN_CHECKPOINT=${TRAIN_CHECKPOINT:-$TRAIN_OUTPUT_DIR/best.pt}
NPZ_PATH=${NPZ_PATH:-$DATA_ROOT/sherlock_sva/meg2sem/sherlock1to9_sva_no_coord_decl5to18_train_minilm_meg2sem_segment3000_fp16.npz}

export PROJECT_ROOT DATA_ROOT
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
export SKIP_ARC_SETUP=${SKIP_ARC_SETUP:-1}
export VAL_NUM_EXAMPLES=${VAL_NUM_EXAMPLES:-512}
export EVAL_NUM_EXAMPLES=${EVAL_NUM_EXAMPLES:-256}
export RETRIEVAL_NUM_EXAMPLES=${RETRIEVAL_NUM_EXAMPLES:-512}
export RETRIEVAL_BATCH_SIZE=${RETRIEVAL_BATCH_SIZE:-256}
export NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-32}

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    export RUN_TAG=${ORACLE_RUN_TAG:-${TRAIN_RUN_TAG}_oracle_eval512}
    export OUTPUT_DIR=${ORACLE_OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/$RUN_TAG}
    export NPZ_PATH
    export ELF_CHECKPOINT=$TRAIN_CHECKPOINT
    export INIT_E2E_CHECKPOINT=$TRAIN_CHECKPOINT
    export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-sherlock_sva_minilm_oracle_overfit_eval}
    export WANDB_RUN_NAME=$RUN_TAG
    exec bash "$PROJECT_ROOT/submit-jobs/sherlock_sva_minilm_oracle_pipe_eval.sbatch"
    ;;
  1)
    export RUN_TAG=${BRAIN_RUN_TAG:-${TRAIN_RUN_TAG}_meg2sem_dt8z7o55_eval512}
    export OUTPUT_DIR=${BRAIN_OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/$RUN_TAG}
    export TRAIN_NPZ=$NPZ_PATH
    export ELF_CHECKPOINT=$TRAIN_CHECKPOINT
    export SEMANTIC_PROJECTOR_CHECKPOINT=$TRAIN_CHECKPOINT
    export MEG2SEM_CHECKPOINT_SELECTION=${MEG2SEM_CHECKPOINT_SELECTION:-pinned}
    export EVAL_ONLY=1
    export TRAIN_MEG2SEM=0
    export TRAIN_SEMANTIC_PROJECTOR=0
    export UNFREEZE_ELF=0
    export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-sherlock_sva_meg2sem_minilm_overfit_eval}
    export WANDB_RUN_NAME=$RUN_TAG
    exec bash "$PROJECT_ROOT/submit-jobs/sherlock_sva_meg2sem_minilm_direct_elf_dt8z7o55.sbatch"
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}" >&2
    exit 2
    ;;
esac
