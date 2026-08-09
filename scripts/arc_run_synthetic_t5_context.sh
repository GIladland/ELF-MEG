#!/usr/bin/env bash
set -euo pipefail

# ARC launcher for first-principles T5-context baselines.
# Exports synthetic T5 latents if missing, optionally runs frozen decode,
# then forwards remaining args to train_npz_t5_bridge_to_elf.py.

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/synthetic-t5-context}
SYNTHETIC_T5_OUTPUT=${SYNTHETIC_T5_OUTPUT:-$DATA_ROOT/synthetic/synthetic_t5_1024_seed7.npz}
SYNTHETIC_COUNT=${SYNTHETIC_COUNT:-1024}
SYNTHETIC_SEED=${SYNTHETIC_SEED:-7}
RUN_FROZEN_DECODE=${RUN_FROZEN_DECODE:-1}
FROZEN_DECODE_EXAMPLES=${FROZEN_DECODE_EXAMPLES:-128}
FROZEN_DECODE_STEPS=${FROZEN_DECODE_STEPS:-32}
FROZEN_RETRIEVAL_EXAMPLES=${FROZEN_RETRIEVAL_EXAMPLES:-$SYNTHETIC_COUNT}
FROZEN_RETRIEVAL_BATCH_SIZE=${FROZEN_RETRIEVAL_BATCH_SIZE:-256}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-synthetic-t5-context}
WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-t5_context_inversion}
INPUT_KEY=${INPUT_KEY:-target_t5_latents}
INPUT_MASK_KEY=${INPUT_MASK_KEY:-}
BRIDGE_TYPE=${BRIDGE_TYPE:-resampler}

source "$(conda info --base)/etc/profile.d/conda.sh"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_ROOT/xdg-cache}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$DATA_ROOT/torch}"
export WANDB_DIR="${WANDB_DIR:-$DATA_ROOT/wandb}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/pip-cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$DATA_ROOT/conda-pkgs}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

mkdir -p "$OUTPUT_DIR" "$(dirname "$SYNTHETIC_T5_OUTPUT")"
cd "$PROJECT_ROOT"

ENV_STAMP_PATH="$TMP_ENV_ROOT/.elf_env_stamp"
if [[ "${SKIP_ARC_SETUP:-0}" == "1" && -d "$TMP_ENV_ROOT/bin" && -f "$ENV_STAMP_PATH" && "$(cat "$ENV_STAMP_PATH")" == "$ENV_STAMP" ]]; then
  echo "Skipping ARC env setup; reusing TMP_ENV_ROOT=$TMP_ENV_ROOT ENV_STAMP=$ENV_STAMP"
elif [[ "${SKIP_ARC_SETUP:-0}" == "1" ]]; then
  echo "SKIP_ARC_SETUP=1 requested, but matching env stamp was not found; falling back to setup: $TMP_ENV_ROOT"
  PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" TMP_ENV_ROOT="$TMP_ENV_ROOT" ENV_STAMP="$ENV_STAMP" \
    bash "$PROJECT_ROOT/scripts/arc_setup_elf_env.sh"
else
  PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" TMP_ENV_ROOT="$TMP_ENV_ROOT" ENV_STAMP="$ENV_STAMP" \
    bash "$PROJECT_ROOT/scripts/arc_setup_elf_env.sh"
fi

conda activate "$TMP_ENV_ROOT"

if [[ ! -f "$SYNTHETIC_T5_OUTPUT" ]]; then
  python scripts/export_synthetic_t5_elf_npz.py \
    --output "$SYNTHETIC_T5_OUTPUT" \
    --count "$SYNTHETIC_COUNT" \
    --seed "$SYNTHETIC_SEED" \
    --batch-size 256 \
    --device cuda
else
  echo "Reusing synthetic T5 NPZ: $SYNTHETIC_T5_OUTPUT"
fi

if [[ "$RUN_FROZEN_DECODE" == "1" ]]; then
  python scripts/decode_elf_context_npz.py \
    --npz-path "$SYNTHETIC_T5_OUTPUT" \
    --output-dir "$OUTPUT_DIR/frozen_decode" \
    --context-key context \
    --context-mask-key context_mask \
    --target-latents-key target_t5_latents \
    --target-mask-key t5_attention_mask \
    --num-examples "$FROZEN_DECODE_EXAMPLES" \
    --target-length 20 \
    --batch-size 16 \
    --num-sampling-steps "$FROZEN_DECODE_STEPS" \
    --cfg-scale 1.0 \
    --self-cond-cfg-scale 1.0 \
    --decode-oracle-target \
    --retrieval-num-examples "$FROZEN_RETRIEVAL_EXAMPLES" \
    --retrieval-batch-size "$FROZEN_RETRIEVAL_BATCH_SIZE" \
    --device cuda
fi

train_args=(
  --npz-path "$SYNTHETIC_T5_OUTPUT"
  --input-key "$INPUT_KEY"
  --target-latents-key target_t5_latents
  --target-mask-key t5_attention_mask
  --target-ids-key t5_input_ids
  --sentence-key sentence
  --bridge-type "$BRIDGE_TYPE"
  --use_wandb
  --wandb_project "$WANDB_PROJECT"
  --wandb_entity "$WANDB_ENTITY"
  --wandb_group "$WANDB_RUN_GROUP"
  --wandb_run_name "$WANDB_RUN_NAME"
  --output_dir "$OUTPUT_DIR/train"
)
if [[ -n "$INPUT_MASK_KEY" ]]; then
  train_args+=(--input-mask-key "$INPUT_MASK_KEY")
fi

python scripts/train_npz_t5_bridge_to_elf.py "${train_args[@]}" "$@"
