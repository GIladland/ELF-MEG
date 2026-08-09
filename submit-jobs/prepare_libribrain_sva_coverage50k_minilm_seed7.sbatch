#!/usr/bin/env bash
#SBATCH --job-name=prep_cov_sva50k
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=submit-jobs/prepare_libribrain_sva_coverage50k_minilm_seed7-%j.out
#SBATCH --error=submit-jobs/prepare_libribrain_sva_coverage50k_minilm_seed7-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}

RUN_TAG=${RUN_TAG:-libribrain_simple_sva_no_coord_decl5to18_sherlock1to9_mocha_timit_moth_valmix}
SVA_TRAIN=${SVA_TRAIN:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_train_minilm_seed7.npz}
SVA_VAL=${SVA_VAL:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_val_minilm_seed7.npz}
SVA_COMBINED=${SVA_COMBINED:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_combined_minilm_seed7.npz}

GUTENBERG_SVA=${GUTENBERG_SVA:-$DATA_ROOT/gutenberg/gutenberg_catalog_en_short5to18_simple_sv_no_coord_minilm_seed7.npz}
CONVERSATION_SVA=${CONVERSATION_SVA:-$DATA_ROOT/conversation/conversation_clean_simple_sv_no_coord_candidates_minilm_seed7.npz}
SELECTED_OUTPUT=${SELECTED_OUTPUT:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_coverage50k_gutenberg_conversation_minilm_seed7.npz}
COMBINED_OUTPUT=${COMBINED_OUTPUT:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_plus_coverage50k_gutenberg_conversation_combined_minilm_seed7.npz}

SELECT_COUNT=${SELECT_COUNT:-50000}
PER_QUERY_CANDIDATES=${PER_QUERY_CANDIDATES:-64}
CANDIDATE_BATCH_SIZE=${CANDIDATE_BATCH_SIZE:-4096}
QUERY_SAMPLE_SIZE=${QUERY_SAMPLE_SIZE:-0}
COSINE_WEIGHT=${COSINE_WEIGHT:-0.40}
LEXICAL_WEIGHT=${LEXICAL_WEIGHT:-0.50}
LENGTH_WEIGHT=${LENGTH_WEIGHT:-0.10}
MIN_LEXICAL_SCORE=${MIN_LEXICAL_SCORE:-0.0}
SEED=${SEED:-7}

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

mkdir -p "$(dirname "$SELECTED_OUTPUT")" "$(dirname "$COMBINED_OUTPUT")" "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"
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

echo "Selecting $SELECT_COUNT coverage-optimized candidates."
python scripts/select_semantic_npz_by_coverage.py \
  --candidate-npz "$GUTENBERG_SVA" \
  --candidate-name gutenberg_sva \
  --candidate-npz "$CONVERSATION_SVA" \
  --candidate-name conversation_sva \
  --query-npz "$SVA_TRAIN" \
  --output "$SELECTED_OUTPUT" \
  --count "$SELECT_COUNT" \
  --per-query-candidates "$PER_QUERY_CANDIDATES" \
  --candidate-batch-size "$CANDIDATE_BATCH_SIZE" \
  --query-sample-size "$QUERY_SAMPLE_SIZE" \
  --blocked-npz "$SVA_COMBINED" \
  --cosine-weight "$COSINE_WEIGHT" \
  --lexical-weight "$LEXICAL_WEIGHT" \
  --length-weight "$LENGTH_WEIGHT" \
  --min-lexical-score "$MIN_LEXICAL_SCORE" \
  --seed "$SEED" \
  --device cuda

echo "Combining SVA train + coverage-selected external candidates with held-out validation."
python scripts/concat_semantic_npzs.py \
  --train-npz "$SVA_TRAIN" \
  --train-npz "$SELECTED_OUTPUT" \
  --val-npz "$SVA_VAL" \
  --output "$COMBINED_OUTPUT" \
  --dedupe-train \
  --dedupe-val \
  --drop-train-overlap-with-val

echo "Prepared coverage dataset: $COMBINED_OUTPUT"
