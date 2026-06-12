#!/usr/bin/env bash
set -euo pipefail

# ARC launcher for the tiny learned-context overfit experiment.
# Uses the known-good node-local env builder and logs to WandB project BrainDiffusion.

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-elf-env}
OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/toy-context-overfit}

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

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" TMP_ENV_ROOT="$TMP_ENV_ROOT" \
  "$PROJECT_ROOT/scripts/arc_setup_elf_env.sh"

conda activate "$TMP_ENV_ROOT"

python scripts/toy_context_overfit.py \
  --use_wandb \
  --wandb_project BrainDiffusion \
  --wandb_run_name toy-context-overfit \
  --output_dir "$OUTPUT_DIR" \
  "$@"
