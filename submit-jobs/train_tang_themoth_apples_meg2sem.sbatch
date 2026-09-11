#!/usr/bin/env bash
#SBATCH --job-name=train_tm_apples
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=submit-jobs/train_tang_themoth_apples_meg2sem-%j.out
#SBATCH --error=submit-jobs/train_tang_themoth_apples_meg2sem-%j.err

set -euo pipefail

MEG2SEM_ROOT=${MEG2SEM_ROOT:-/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
OUTPUT_ROOT=${OUTPUT_ROOT:-$DATA_ROOT/tang_themoth_apples_exact384_v2}
PACKED_ROOT=${PACKED_ROOT:-$OUTPUT_ROOT/packed}
SEGMENT_MS=${SEGMENT_MS:-20000}
PREFIX=${PREFIX:-tang_themoth_exact384_minilm384_seg${SEGMENT_MS}}
RUN_NAME=${RUN_NAME:-tang_themoth_apples_exact384_seg20s_brainmagick_seed60_v2}

train_npz=${TRAIN_NPZ:-$PACKED_ROOT/${PREFIX}_train_fp16.npz}
val_npz=${VAL_NPZ:-$PACKED_ROOT/${PREFIX}_val_fp16.npz}
for path in "$train_npz" "$val_npz"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing packed split: $path" >&2
    exit 2
  fi
done

cd "$MEG2SEM_ROOT"
# MEG2SEM otherwise silently falls back to its unrelated 1536-D Sherlock
# test set after fitting. Reuse validation only for that post-fit code path;
# the locked birthofanation pack remains completely absent from this job.
env \
  WANDB_PROJECT=${WANDB_PROJECT:-braindiffusion_brain2sem} \
  WANDB_JOB_TYPE=train \
  WANDB_GROUP=tang_themoth_apples_exact384_minilm \
  WANDB_TAGS=${WANDB_TAGS:-clean,apples_to_apples,lb2,podcast,themoth,minilm,exact384,no_test_during_selection} \
  CHECKPOINT_MONITOR=val_nDCG_epoch \
  CHECKPOINT_MODE=max \
  RUN_NAME="$RUN_NAME" \
  RUN_BOOKS=1 \
  SEED=${SEED:-60} \
  PACKED_NPZ="$train_npz" \
  VAL_PACKED_NPZ="$val_npz" \
  TEST_PACKED_NPZ="$val_npz" \
  SEGMENT_MS="$SEGMENT_MS" \
  EMBEDDING_TYPE=MINILM \
  EMBEDDING_DIM=384 \
  LOSS_TYPE=siglip \
  CONTRASTIVE_WEIGHT=${CONTRASTIVE_WEIGHT:-5} \
  COSINE_WEIGHT=${COSINE_WEIGHT:-1} \
  VICREG_WEIGHT=${VICREG_WEIGHT:-0} \
  NORM_LOSS_WEIGHT=${NORM_LOSS_WEIGHT:-0} \
  LEARNING_RATE=${LEARNING_RATE:-0.0003} \
  DROPOUT=${DROPOUT:-0.1} \
  BATCH_SIZE=${BATCH_SIZE:-8} \
  ACCUMULATE_GRAD_BATCHES=${ACCUMULATE_GRAD_BATCHES:-4} \
  MAX_EPOCHS=${MAX_EPOCHS:-80} \
  PRECISION=${PRECISION:-bf16-mixed} \
  UNIQUE_WORD_BATCHES=1 \
  USE_PROJECTION_HEAD=1 \
  MODEL_NORM=batch \
  AGGREGATION=${AGGREGATION:-attention} \
  ATTENTION_HEADS=4 \
  BRAIN_MODEL_ARCH=brainmagick \
  BRAIN_CONV_CHANNELS=320 \
  BRAIN_CONV_LAYERS=10 \
  BRAIN_CONV_DILATION_GROWTH=2 \
  BATCH_SAMPLER_KEY=meg_duration \
  BATCH_SAMPLER_POOL_SIZE=250 \
  CONTRASTIVE_INIT_TEMPERATURE=5 \
  CONTRASTIVE_INIT_BIAS=-10 \
  NOTES="${NOTES:-Leakage-free apples-to-apples MEG2SEM. Labels come from the exact same delayed-MiniLM exact384 oracle used by the matched fMRI and shared ELF runs. Train on 26 TheMoth stories; select only on thatthingonmyarm, leavingbaghdad, and howtodraw; birthofanation MEG is absent from fitting and selection.}" \
  bash "$MEG2SEM_ROOT/task-submit-sherlock-sva-ada002.sh"
