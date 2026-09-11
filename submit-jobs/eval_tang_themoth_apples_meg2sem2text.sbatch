#!/usr/bin/env bash
#SBATCH --job-name=eval_tm_apples
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --array=0-1
#SBATCH --output=submit-jobs/eval_tang_themoth_apples_meg2sem2text-%A_%a.out
#SBATCH --error=submit-jobs/eval_tang_themoth_apples_meg2sem2text-%A_%a.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
MEG2SEM_ROOT=${MEG2SEM_ROOT:-/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
OUTPUT_ROOT=${OUTPUT_ROOT:-$DATA_ROOT/tang_themoth_apples_exact384_v2}
PACKED_ROOT=${PACKED_ROOT:-$OUTPUT_ROOT/packed}
SEGMENT_MS=${SEGMENT_MS:-20000}
PREFIX=${PREFIX:-tang_themoth_exact384_minilm384_seg${SEGMENT_MS}}
MEG2SEM_RUN_NAME=${MEG2SEM_RUN_NAME:-tang_themoth_apples_exact384_seg20s_brainmagick_seed60_v2}
EVAL_LABEL=${EVAL_LABEL:-exact384_direct_meg2sem2text_fixedelf_steps16_cfg0p5_seed101}
ELF_RUN=${ELF_RUN:-fmri_minilm_knowntext_exact384_all12098_testfirst_full_k64_ep100_b64_seed49_20260829}
ELF_CHECKPOINT=${ELF_CHECKPOINT:-/data/engs-pnpl/glandau/elf-runs/$ELF_RUN/best.pt}
eval_variant=${EVAL_VARIANT:-$SLURM_ARRAY_TASK_ID}

case "$eval_variant" in
  0) split=val ;;
  1) split=test ;;
  *) echo "Unsupported EVAL_VARIANT=$eval_variant" >&2; exit 2 ;;
esac

train_npz=$PACKED_ROOT/${PREFIX}_train_fp16.npz
eval_npz=$PACKED_ROOT/${PREFIX}_${split}_fp16.npz
for path in "$train_npz" "$eval_npz" "$ELF_CHECKPOINT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required artifact: $path" >&2
    exit 2
  fi
done

cd "$PROJECT_ROOT"
MEG2SEM_CHECKPOINT=${MEG2SEM_CHECKPOINT:-$(python scripts/select_named_meg2sem_checkpoint.py \
  --meg2sem-root "$MEG2SEM_ROOT" \
  --run-name "$MEG2SEM_RUN_NAME" \
  --metric val_nDCG_epoch \
  --mode max \
  --output path)}

# The runner inspects TRAIN_NPZ even in eval-only mode. Point it at the same
# held-out pack to avoid decompressing the 13.5-GB training MEG array; no
# optimizer or fitting path is enabled below.
export TRAIN_NPZ="$eval_npz"
export VAL_NPZ="$eval_npz"
export MEG2SEM_CHECKPOINT
export ELF_CHECKPOINT
export SEMANTIC_PROJECTOR_CHECKPOINT="$ELF_CHECKPOINT"
export INIT_E2E_CHECKPOINT=
export MEG2SEM_OUTPUT_NORMALIZATION=auto
export MEG2SEM_PREDICTION_MODE=${MEG2SEM_PREDICTION_MODE:-full_ada}
export MEG2SEM_RESIDUAL_COMPONENT_NPZ=${MEG2SEM_RESIDUAL_COMPONENT_NPZ:-}
export MEG2SEM_RESIDUAL_MEAN_KEY=${MEG2SEM_RESIDUAL_MEAN_KEY:-train_mean}
export MEG2SEM_RESIDUAL_TARGET_SCALE=${MEG2SEM_RESIDUAL_TARGET_SCALE:-1.0}
export EVAL_ONLY=1
export TRAIN_MEG2SEM=0
export TRAIN_SEMANTIC_PROJECTOR=0
export UNFREEZE_ELF=0
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=0
export RETRIEVAL_NUM_EXAMPLES=0
export EVAL_START_INDEX=0
export RETRIEVAL_START_INDEX=0
# Attention over 5,000 MEG samples is quadratic in time. A retrieval batch of
# eight attempted a 41-GiB allocation on an L40S, so keep every path that calls
# the frozen MEG encoder at batch one. This is numerically equivalent and only
# affects wall time.
export BATCH_SIZE=${BATCH_SIZE:-1}
export RETRIEVAL_BATCH_SIZE=${RETRIEVAL_BATCH_SIZE:-128}
export INTERFACE_MONITOR_BATCH_SIZE=${INTERFACE_MONITOR_BATCH_SIZE:-0}

# Fixed to the same validation-selected sampler used for the fMRI known-text
# comparison.  There is no MEG-test-specific decoding selection.
export NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-16}
export CFG_SCALE=${CFG_SCALE:-0.5}
export SELF_COND_CFG_SCALE=1.0
export SEED=${SEED:-101}
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export RUN_TAG=${RUN_TAG:-tang_themoth_APPLES_${split}_${EVAL_LABEL}_20260830}
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export WANDB_RUN_GROUP=tang_themoth_apples_meg2sem2text_20260830
export WANDB_RUN_NAME=$RUN_TAG

exec bash "$PROJECT_ROOT/submit-jobs/sherlock_sva_meg2sem_minilm_direct_elf_dt8z7o55.sbatch"
