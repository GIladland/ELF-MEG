#!/usr/bin/env bash
set -euo pipefail

# ARC launcher for the tiny LibriBrain MEG-context overfit experiment.

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/meg-context-overfit}
LIBRIBRAIN_DATA_PATH=${LIBRIBRAIN_DATA_PATH:-}
LIBRIBRAIN_SEMANTIC_DATA_PATH=${LIBRIBRAIN_SEMANTIC_DATA_PATH:-}
PNPL_ROOT=${PNPL_ROOT:-/data/engs-pnpl/glandau/MEG2SEM/PNPL/pnpl}

if [[ -z "$LIBRIBRAIN_DATA_PATH" ]]; then
  echo "Set LIBRIBRAIN_DATA_PATH to the LibriBrain dataset root before launching." >&2
  exit 1
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

cmd=(
  python scripts/meg_context_overfit.py
  --data-path "$LIBRIBRAIN_DATA_PATH"
  --pnpl-root "$PNPL_ROOT"
  --use_wandb
  --wandb_project BrainDiffusion
  --wandb_run_name meg-context-overfit
  --output_dir "$OUTPUT_DIR"
)

if [[ -n "$LIBRIBRAIN_SEMANTIC_DATA_PATH" ]]; then
  cmd+=(--semantic-data-path "$LIBRIBRAIN_SEMANTIC_DATA_PATH")
fi

cmd+=("$@")
"${cmd[@]}"
