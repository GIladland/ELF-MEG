#!/usr/bin/env bash
set -euo pipefail

# ARC launcher for fixed semantic-vector NPZ -> ELF text generation runs.
# Optionally exports a synthetic local-embedding NPZ before training.

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/npz-semantic-to-elf}
NPZ_PATH=${NPZ_PATH:-}
SYNTHETIC_OUTPUT=${SYNTHETIC_OUTPUT:-}
SYNTHETIC_COUNT=${SYNTHETIC_COUNT:-0}
SYNTHETIC_SEED=${SYNTHETIC_SEED:-0}
SYNTHETIC_EMBEDDING_MODEL=${SYNTHETIC_EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}
SYNTHETIC_INPUT_PREFIX=${SYNTHETIC_INPUT_PREFIX:-}
SYNTHETIC_POOLING=${SYNTHETIC_POOLING:-mean}
SYNTHETIC_BATCH_SIZE=${SYNTHETIC_BATCH_SIZE:-256}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-npz-semantic-to-elf}
WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-conditioning_ladder}

if ! command -v conda >/dev/null 2>&1; then
  for conda_sh in \
    "${CONDA_SH:-}" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/data/engs-pnpl/glandau/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -n "$conda_sh" && -f "$conda_sh" ]]; then
      # Make nested Slurm submissions independent of the login shell PATH.
      source "$conda_sh"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found; set CONDA_SH to the conda.sh path before launching." >&2
  exit 127
fi

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

mkdir -p "$OUTPUT_DIR"
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

if [[ "$SYNTHETIC_COUNT" != "0" ]]; then
  if [[ -z "$SYNTHETIC_OUTPUT" ]]; then
    SYNTHETIC_OUTPUT="$DATA_ROOT/synthetic/synthetic_minilm_${SYNTHETIC_COUNT}_seed${SYNTHETIC_SEED}.npz"
  fi
  mkdir -p "$(dirname "$SYNTHETIC_OUTPUT")"
  if [[ ! -f "$SYNTHETIC_OUTPUT" ]]; then
    python scripts/export_synthetic_sentence_embeddings.py \
      --output "$SYNTHETIC_OUTPUT" \
      --count "$SYNTHETIC_COUNT" \
      --seed "$SYNTHETIC_SEED" \
      --embedding-model-name "$SYNTHETIC_EMBEDDING_MODEL" \
      --input-prefix "$SYNTHETIC_INPUT_PREFIX" \
      --pooling "$SYNTHETIC_POOLING" \
      --batch-size "$SYNTHETIC_BATCH_SIZE" \
      --device cuda
  else
    echo "Reusing existing synthetic NPZ: $SYNTHETIC_OUTPUT"
  fi
  NPZ_PATH="$SYNTHETIC_OUTPUT"
fi

if [[ -z "$NPZ_PATH" ]]; then
  echo "Set NPZ_PATH or SYNTHETIC_COUNT before launching." >&2
  exit 1
fi

python scripts/train_npz_semantic_to_elf.py \
  --npz-path "$NPZ_PATH" \
  --use_wandb \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_group "$WANDB_RUN_GROUP" \
  --wandb_run_name "$WANDB_RUN_NAME" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
