#!/usr/bin/env bash
#SBATCH --job-name=train_tm_resid
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=submit-jobs/train_tang_themoth_apples_meg2sem_residual-%j.out
#SBATCH --error=submit-jobs/train_tang_themoth_apples_meg2sem_residual-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
COMPONENT_NPZ=${COMPONENT_NPZ:-/data/engs-pnpl/glandau/elf-cache/tang_themoth_apples_exact384_v2/tang_themoth_exact384_minilm384_train_mean.npz}

if [[ ! -f "$COMPONENT_NPZ" ]]; then
  echo "Missing train-only semantic component: $COMPONENT_NPZ" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
env \
  RUN_NAME=${RUN_NAME:-tang_themoth_apples_exact384_seg20s_brainmagick_residual_queue_seed61_v2} \
  SEED=${SEED:-61} \
  MAX_EPOCHS=${MAX_EPOCHS:-80} \
  ADA_RESIDUAL_TARGET_MODE=mean_only \
  ADA_RESIDUAL_TARGET_MU_NPZ="$COMPONENT_NPZ" \
  ADA_RESIDUAL_TARGET_SCALE=${ADA_RESIDUAL_TARGET_SCALE:-1.0} \
  USE_NEGATIVE_QUEUE=1 \
  NEGATIVE_QUEUE_SIZE=${NEGATIVE_QUEUE_SIZE:-4096} \
  CONTRASTIVE_WEIGHT=${CONTRASTIVE_WEIGHT:-2.0} \
  COSINE_WEIGHT=${COSINE_WEIGHT:-1.0} \
  RESIDUAL_TARGET_COSINE_WEIGHT=${RESIDUAL_TARGET_COSINE_WEIGHT:-2.0} \
  RESIDUAL_TARGET_MSE_WEIGHT=${RESIDUAL_TARGET_MSE_WEIGHT:-10.0} \
  RESIDUAL_TARGET_CONTRASTIVE_WEIGHT=${RESIDUAL_TARGET_CONTRASTIVE_WEIGHT:-3.0} \
  NOTES="Leakage-free Tang-TheMoth residual MEG2SEM using the same exact384 labels as the matched fMRI/shared-ELF system. The MiniLM mean and all negative-queue items come only from 26 training stories. Selection uses only thatthingonmyarm, leavingbaghdad, and howtodraw; locked birthofanation is absent." \
  bash submit-jobs/train_tang_themoth_apples_meg2sem.sbatch
