#!/usr/bin/env bash
#SBATCH --job-name=prep_gut_sva100k
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --output=submit-jobs/prepare_gutenberg_sva_matched100k_minilm_seed7-%j.out
#SBATCH --error=submit-jobs/prepare_gutenberg_sva_matched100k_minilm_seed7-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}

GUTENBERG_CANDIDATE=${GUTENBERG_CANDIDATE:-$DATA_ROOT/gutenberg/gutenberg_catalog_en_short5to18_content3m_minilm_seed7.npz}
GUTENBERG_SVA=${GUTENBERG_SVA:-$DATA_ROOT/gutenberg/gutenberg_catalog_en_short5to18_simple_sv_no_coord_minilm_seed7.npz}
GUTENBERG_SELECTED=${GUTENBERG_SELECTED:-$DATA_ROOT/gutenberg/gutenberg_catalog_en_short5to18_simple_sv_no_coord_matched100k_by_libribrain_sva_train_minilm_seed7.npz}

RUN_TAG=${RUN_TAG:-libribrain_simple_sva_no_coord_decl5to18_sherlock1to9_mocha_timit_moth_valmix}
SVA_TRAIN=${SVA_TRAIN:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_train_minilm_seed7.npz}
SVA_VAL=${SVA_VAL:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_val_minilm_seed7.npz}
SVA_COMBINED=${SVA_COMBINED:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_combined_minilm_seed7.npz}
OUTPUT_COMBINED=${OUTPUT_COMBINED:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_plus_gutenberg_svamatch100k_combined_minilm_seed7.npz}

SELECT_COUNT=${SELECT_COUNT:-100000}
QUERY_SAMPLE_SIZE=${QUERY_SAMPLE_SIZE:-0}
PER_QUERY_CANDIDATES=${PER_QUERY_CANDIDATES:-32}
CANDIDATE_BATCH_SIZE=${CANDIDATE_BATCH_SIZE:-4096}
SEED=${SEED:-7}

source "$(conda info --base)/etc/profile.d/conda.sh"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_ROOT/xdg-cache}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$DATA_ROOT/torch}"
export WANDB_DIR="${WANDB_DIR:-$DATA_ROOT/wandb}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$DATA_ROOT/wandb-data}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$DATA_ROOT/wandb-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/pip-cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$DATA_ROOT/conda-pkgs}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TMPDIR="${TMPDIR:-$DATA_ROOT/tmp}"

mkdir -p \
  "$(dirname "$GUTENBERG_SVA")" \
  "$(dirname "$GUTENBERG_SELECTED")" \
  "$(dirname "$OUTPUT_COMBINED")" \
  "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"
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

if [[ ! -f "$GUTENBERG_SVA" ]]; then
  (
    flock -x 9
    if [[ ! -f "$GUTENBERG_SVA" ]]; then
      echo "Filtering Gutenberg candidates to simple_sv_no_coord: $GUTENBERG_SVA"
      python scripts/filter_semantic_npz_sentences.py \
        --input "$GUTENBERG_CANDIDATE" \
        --output "$GUTENBERG_SVA" \
        --filter-mode simple_sv_no_coord \
        --min-words 5 \
        --max-words 18 \
        --min-alpha-fraction 0.75 \
        --min-content-words 2 \
        --dedupe
    fi
  ) 9>"$GUTENBERG_SVA.lock"
else
  echo "Using existing filtered Gutenberg SVA NPZ: $GUTENBERG_SVA"
fi

if [[ ! -f "$GUTENBERG_SELECTED" ]]; then
  (
    flock -x 9
    if [[ ! -f "$GUTENBERG_SELECTED" ]]; then
      echo "Selecting $SELECT_COUNT Gutenberg SVA rows matched to LibriBrain/Sherlock SVA train."
      python scripts/select_semantic_npz_by_similarity.py \
        --candidate-npz "$GUTENBERG_SVA" \
        --query-npz "$SVA_TRAIN" \
        --output "$GUTENBERG_SELECTED" \
        --count "$SELECT_COUNT" \
        --selection-mode query_balanced \
        --query-sample-size "$QUERY_SAMPLE_SIZE" \
        --per-query-candidates "$PER_QUERY_CANDIDATES" \
        --candidate-batch-size "$CANDIDATE_BATCH_SIZE" \
        --blocked-npz "$SVA_COMBINED" \
        --seed "$SEED" \
        --device cuda
    fi
  ) 9>"$GUTENBERG_SELECTED.lock"
else
  echo "Using existing selected Gutenberg SVA NPZ: $GUTENBERG_SELECTED"
fi

echo "Combining current SVA train + matched Gutenberg SVA 100k with held-out SVA validation."
python scripts/concat_semantic_npzs.py \
  --train-npz "$SVA_TRAIN" \
  --train-npz "$GUTENBERG_SELECTED" \
  --val-npz "$SVA_VAL" \
  --output "$OUTPUT_COMBINED" \
  --dedupe-train \
  --dedupe-val \
  --drop-train-overlap-with-val

echo "Prepared combined dataset: $OUTPUT_COMBINED"
