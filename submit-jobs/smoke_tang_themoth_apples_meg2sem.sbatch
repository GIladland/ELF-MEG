#!/usr/bin/env bash
#SBATCH --job-name=smoke_tm_apples
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=submit-jobs/smoke_tang_themoth_apples_meg2sem-%j.out
#SBATCH --error=submit-jobs/smoke_tang_themoth_apples_meg2sem-%j.err

set -euo pipefail

MEG2SEM_ROOT=${MEG2SEM_ROOT:-/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
packed=$DATA_ROOT/tang_themoth_apples_v1/packed/tang_themoth_middle10_minilm384_seg20000_val_fp16.npz

test -f "$packed"
cd "$MEG2SEM_ROOT"

# Configuration smoke only: use the 110-row validation pack as both sides,
# discard its checkpoint, and never expose the locked test pack.
env \
  WANDB_PROJECT=braindiffusion_brain2sem \
  WANDB_JOB_TYPE=smoke \
  WANDB_GROUP=tang_themoth_apples_middle10_minilm384_smoke \
  WANDB_TAGS=smoke,discard,no_test \
  RUN_NAME=tang_themoth_apples_middle10_minilm384_seg20s_SMOKE_DISCARD \
  RUN_BOOKS=1 \
  SEED=58 \
  PACKED_NPZ="$packed" \
  VAL_PACKED_NPZ="$packed" \
  TEST_PACKED_NPZ="$packed" \
  SEGMENT_MS=20000 \
  EMBEDDING_TYPE=MINILM \
  EMBEDDING_DIM=384 \
  LOSS_TYPE=siglip \
  CONTRASTIVE_WEIGHT=5 \
  COSINE_WEIGHT=1 \
  VICREG_WEIGHT=0 \
  NORM_LOSS_WEIGHT=0 \
  LEARNING_RATE=0.0003 \
  DROPOUT=0.1 \
  BATCH_SIZE=8 \
  ACCUMULATE_GRAD_BATCHES=4 \
  MAX_EPOCHS=1 \
  PRECISION=16-mixed \
  UNIQUE_WORD_BATCHES=1 \
  USE_PROJECTION_HEAD=1 \
  MODEL_NORM=batch \
  AGGREGATION=attention \
  ATTENTION_HEADS=4 \
  BRAIN_MODEL_ARCH=brainmagick \
  BRAIN_CONV_CHANNELS=320 \
  BRAIN_CONV_LAYERS=10 \
  BRAIN_CONV_DILATION_GROWTH=2 \
  BATCH_SAMPLER_KEY=meg_duration \
  BATCH_SAMPLER_POOL_SIZE=250 \
  CONTRASTIVE_INIT_TEMPERATURE=5 \
  CONTRASTIVE_INIT_BIAS=-10 \
  NOTES="Discarded one-epoch configuration smoke for the Tang/TheMoth apples-to-apples run; locked test NPZ is absent." \
  bash "$MEG2SEM_ROOT/task-submit-sherlock-sva-ada002.sh"
