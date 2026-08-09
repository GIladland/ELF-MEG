#!/usr/bin/env bash
#SBATCH --job-name=an_cov_sets
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=submit-jobs/analyze_libribrain_sva_coverage_sets-%j.out
#SBATCH --error=submit-jobs/analyze_libribrain_sva_coverage_sets-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
OUTPUT_JSON=${OUTPUT_JSON:-$DATA_ROOT/coverage/libribrain_sva_coverage_sets.json}
INCLUDE_COVERAGE50K=${INCLUDE_COVERAGE50K:-0}

source "$(conda info --base)/etc/profile.d/conda.sh"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_ROOT/xdg-cache}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${TORCH_HOME:-$DATA_ROOT/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/pip-cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$DATA_ROOT/conda-pkgs}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TMPDIR="${TMPDIR:-$DATA_ROOT/tmp}"

mkdir -p "$(dirname "$OUTPUT_JSON")" "$TMPDIR"
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

ARGS=(
  --npz "$DATA_ROOT/synthetic/synthetic_minilm_11264_seed7.npz"
  --name synthetic_11k_template
  --npz "$DATA_ROOT/synthetic/synthetic_minilm_103424_seed7_fixedval10240_1024.npz"
  --name synthetic_103k_template
  --npz "$DATA_ROOT/libribrain_simple/libribrain_simple_sva_no_coord_decl5to18_sherlock1to9_mocha_timit_moth_valmix_combined_minilm_seed7.npz"
  --name libribrain_sva_base13k
  --npz "$DATA_ROOT/libribrain_simple/libribrain_simple_sva_no_coord_decl5to18_sherlock1to9_mocha_timit_moth_valmix_plus_gutenberg_svamatch100k_combined_minilm_seed7.npz"
  --name libribrain_sva_plus_gutenberg100k
  --npz "$DATA_ROOT/libribrain_simple/libribrain_simple_sva_no_coord_decl5to18_sherlock1to9_mocha_timit_moth_valmix_plus_conversation_svamatch100k_combined_minilm_seed7.npz"
  --name libribrain_sva_plus_conversation55k
  --npz "$DATA_ROOT/sherlock/sherlock_books1to9_short5to18_train_sherlock12_short5to18_val_minilm_seed7.npz"
  --name sherlock_short5to18
)

if [[ "$INCLUDE_COVERAGE50K" == "1" ]]; then
  ARGS+=(
    --npz "$DATA_ROOT/libribrain_simple/libribrain_simple_sva_no_coord_decl5to18_sherlock1to9_mocha_timit_moth_valmix_plus_coverage50k_gutenberg_conversation_combined_minilm_seed7.npz"
    --name libribrain_sva_plus_coverage50k
  )
fi

python scripts/analyze_npz_coverage.py \
  "${ARGS[@]}" \
  --val-num-examples 1024 \
  --train-query-sample-size 4096 \
  --batch-size 128 \
  --seed 7 \
  --device cuda \
  --output-json "$OUTPUT_JSON"
