#!/usr/bin/env bash
set -euo pipefail

# ARC LibriBrain PNPL smoke test.
# Reuses the canonical ARC ELF setup script so the node-local env is rebuilt
# from requirements.txt, while all large caches stay on /data.

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-elf-env}
LIBRIBRAIN_ROOT=${LIBRIBRAIN_ROOT:-/data/engs-pnpl/glandau/MEG2SEM/LibriBrain}
PNPL_ROOT=${PNPL_ROOT:-/data/engs-pnpl/glandau/MEG2SEM/PNPL/pnpl}
BOOKS=${BOOKS:-1}
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_BATCHES=${NUM_BATCHES:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
SEGMENT_MS=${SEGMENT_MS:-3000}

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

cd "$PROJECT_ROOT"

PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" TMP_ENV_ROOT="$TMP_ENV_ROOT" \
  "$PROJECT_ROOT/scripts/arc_setup_elf_env.sh"

conda activate "$TMP_ENV_ROOT"

echo "HOST: $(hostname)"
echo "LIBRIBRAIN_ROOT=$LIBRIBRAIN_ROOT"
echo "PNPL_ROOT=$PNPL_ROOT"
echo "BOOKS=$BOOKS"
python --version

read -r -a BOOK_ARGS <<< "$BOOKS"

python scripts/probe_libribrain_loader.py \
  --data-path "$LIBRIBRAIN_ROOT" \
  --pnpl-root "$PNPL_ROOT" \
  --books "${BOOK_ARGS[@]}" \
  --batch-size "$BATCH_SIZE" \
  --num-batches "$NUM_BATCHES" \
  --num-workers "$NUM_WORKERS" \
  --segment-ms "$SEGMENT_MS"
