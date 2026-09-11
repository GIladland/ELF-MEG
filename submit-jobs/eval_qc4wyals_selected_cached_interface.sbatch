#!/usr/bin/env bash
#SBATCH --job-name=qc4w_cacheeval
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --array=0-4%4
#SBATCH --output=submit-jobs/eval_qc4wyals_selected_cached_interface-%A_%a.out
#SBATCH --error=submit-jobs/eval_qc4wyals_selected_cached_interface-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
interface_dir=$DATA_ROOT/tang_themoth_qc4wyals_ada_v2/interface
eval_split=${EVAL_SPLIT:-val}
case "$eval_split" in
  val)
    eval_rows=110
    source_npz=$interface_dir/qc4wyals_val_predicted_ada002.npz
    ;;
  test)
    eval_rows=26
    source_npz=$interface_dir/qc4wyals_test_predicted_ada002.npz
    ;;
  *)
    echo "EVAL_SPLIT must be val or test, got: $eval_split" >&2
    exit 2
    ;;
esac
source_npz=${SOURCE_NPZ_OVERRIDE:-$source_npz}

: "${SELECTED_CHECKPOINT:?Set SELECTED_CHECKPOINT to one validation-selected interface checkpoint}"
[[ -f "$SELECTED_CHECKPOINT" ]] || { echo "Missing $SELECTED_CHECKPOINT" >&2; exit 2; }

seeds=(49 101 202 303 404)
if [[ -n ${SEED_OVERRIDE:-} ]]; then
  export SEED=$SEED_OVERRIDE
else
  export SEED=${seeds[$SLURM_ARRAY_TASK_ID]}
fi
selected_tag=${SELECTED_TAG:-$(basename "$SELECTED_CHECKPOINT" .pt)}
export RUN_TAG=eval_qc4wyals_${eval_split}${eval_rows}_${selected_tag}_seed${SEED}_20260901
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export SOURCE_NPZ=$source_npz
export NPZ_PATH=$source_npz
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export MODEL=ELF-B
export CHECKPOINT_PATH=embedded-language-flows/ELF-B-owt-torch
export INIT_E2E_CHECKPOINT=$SELECTED_CHECKPOINT
export INIT_E2E_ADAPTER_MISMATCH=${INIT_E2E_ADAPTER_MISMATCH:-error}
export SEMANTIC_ADAPTER_KIND=${SEMANTIC_ADAPTER_KIND:-flat}
export SEMANTIC_INPUT_PROJECTION_DIM=0
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export ADAPTER_DROPOUT=0.0
export FREEZE_ELF=1
export FREEZE_SEMANTIC_ADAPTER=1
export ELF_LORA_RANK=${ELF_LORA_RANK:-0}
export ELF_LORA_ALPHA=${ELF_LORA_ALPHA:-16}
export ELF_LORA_DROPOUT=${ELF_LORA_DROPOUT:-0.05}
export ELF_LORA_TARGETS=${ELF_LORA_TARGETS:-attention,mlp}
export ELF_LORA_LAST_N_BLOCKS=${ELF_LORA_LAST_N_BLOCKS:--1}
export EVAL_ONLY=1
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=$eval_rows
export RETRIEVAL_NUM_EXAMPLES=$eval_rows
export GENERATION_T5_RETRIEVAL=1
export EPOCHS=0
export BATCH_SIZE=22
export EVAL_EVERY=1
export RETRIEVAL_EVAL_EVERY=1
export NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-32}
export TRAIN_TIMESTEP_STEPS=${TRAIN_TIMESTEP_STEPS:-32}
export CFG_SCALE=${CFG_SCALE:-1.0}
export SELF_COND_CFG_SCALE=${SELF_COND_CFG_SCALE:-1.0}
export USE_WANDB=0
export SAVE_FINAL_CHECKPOINT=0
export SAVE_EVAL_CHECKPOINTS=0
# The cache metadata includes the source NPZ.  Validation arrays that evaluate
# different transformed interface packs must not concurrently rewrite one
# shared `.tmp` path; use a seed-local cache to make every array task atomic.
export TARGET_LATENTS_CACHE=${TARGET_LATENTS_CACHE_OVERRIDE:-$interface_dir/qc4wyals_${eval_split}${eval_rows}_${selected_tag}_seed${SEED}_t5small_latents_fp16.npy}
export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
