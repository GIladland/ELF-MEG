#!/usr/bin/env bash
#SBATCH --job-name=prep_sh_ret5
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=submit-jobs/prepare_sherlock_sva_retrieval_t5top5_minilm_seed7-%j.out
#SBATCH --error=submit-jobs/prepare_sherlock_sva_retrieval_t5top5_minilm_seed7-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}

SHERLOCK_DIR=${SHERLOCK_DIR:-$DATA_ROOT/sherlock_sva}
SOURCE_TAG=${SOURCE_TAG:-sherlock1to9_sva_no_coord_decl5to18_plus_trainaug_v1_5x_to_sherlock12_sva_no_coord_decl5to18}
RETRIEVAL_TAG=${RETRIEVAL_TAG:-sherlock1to9_sva_no_coord_decl5to18_plus_trainaug_v1_5x_retrievalt5top5_to_sherlock12_sva_no_coord_decl5to18}
SOURCE_NPZ=${SOURCE_NPZ:-$SHERLOCK_DIR/${SOURCE_TAG}_combined_minilm_seed7.npz}
RETRIEVAL_NPZ=${RETRIEVAL_NPZ:-$SHERLOCK_DIR/${RETRIEVAL_TAG}_combined_minilm_seed7.npz}

source "$(conda info --base)/etc/profile.d/conda.sh"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_ROOT/xdg-cache}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${TORCH_HOME:-$DATA_ROOT/torch}"
export WANDB_DIR="${WANDB_DIR:-$DATA_ROOT/wandb}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$DATA_ROOT/wandb-data}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$DATA_ROOT/wandb-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/pip-cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$DATA_ROOT/conda-pkgs}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TMPDIR="${TMPDIR:-$DATA_ROOT/tmp}"

mkdir -p "$SHERLOCK_DIR" "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"
cd "$PROJECT_ROOT"

ENV_STAMP_PATH="$TMP_ENV_ROOT/.elf_env_stamp"
if [[ "${SKIP_ARC_SETUP:-1}" == "1" && -d "$TMP_ENV_ROOT/bin" && -f "$ENV_STAMP_PATH" && "$(cat "$ENV_STAMP_PATH")" == "$ENV_STAMP" ]]; then
  echo "Skipping ARC env setup; reusing TMP_ENV_ROOT=$TMP_ENV_ROOT ENV_STAMP=$ENV_STAMP"
else
  echo "Preparing ARC env at $TMP_ENV_ROOT"
  PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" TMP_ENV_ROOT="$TMP_ENV_ROOT" ENV_STAMP="$ENV_STAMP" \
    bash "$PROJECT_ROOT/scripts/arc_setup_elf_env.sh"
fi

conda activate "$TMP_ENV_ROOT"

if [[ ! -f "$SOURCE_NPZ" ]]; then
  echo "Missing source NPZ: $SOURCE_NPZ" >&2
  exit 1
fi

if [[ ! -f "$RETRIEVAL_NPZ" ]]; then
  python scripts/build_retrieval_augmented_npz.py \
    --npz "$SOURCE_NPZ" \
    --output "$RETRIEVAL_NPZ" \
    --neighbor-top-k 5 \
    --retrieval-batch-size 256 \
    --t5-batch-size 128 \
    --encoder-model-name t5-small \
    --latent-mean 0.0 \
    --latent-std 0.2 \
    --device cuda \
    --exclude-identical-text \
    --include-scores
fi

echo "Prepared retrieval-augmented NPZ:"
echo "$RETRIEVAL_NPZ"
echo "${RETRIEVAL_NPZ%.npz}.summary.json"
