#!/usr/bin/env bash
#SBATCH --job-name=sh_cov_hybrid
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-18:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=220G
#SBATCH --output=submit-jobs/prepare_sherlock_sva_clean_hybrids-%j.out
#SBATCH --error=submit-jobs/prepare_sherlock_sva_clean_hybrids-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
SUITE_DIR=${SUITE_DIR:-$DATA_ROOT/sherlock_sva/coverage_suite_v1}
TRAIN=$DATA_ROOT/sherlock_sva/sherlock1to9_sva_no_coord_decl5to18_train_minilm_seed7.npz
VAL=$DATA_ROOT/sherlock_sva/sherlock12_sva_no_coord_decl5to18_val_minilm_seed7.npz
TRAIN_NEIGH=$SUITE_DIR/train_gramneigh64x_minilm_seed7.npz
EXTERNAL_NEIGH=$SUITE_DIR/clean_balanced_top20k_gramneigh32x_minilm_seed7.npz

source "$(conda info --base)/etc/profile.d/conda.sh"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$DATA_ROOT/xdg-cache}
export HF_HOME=${HF_HOME:-$DATA_ROOT/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export TMPDIR=${TMPDIR:-$DATA_ROOT/tmp}

mkdir -p "$SUITE_DIR" "$TMPDIR"
cd "$PROJECT_ROOT"
ENV_STAMP_PATH="$TMP_ENV_ROOT/.elf_env_stamp"
if [[ "${SKIP_ARC_SETUP:-1}" == "1" && -d "$TMP_ENV_ROOT/bin" && -f "$ENV_STAMP_PATH" && "$(cat "$ENV_STAMP_PATH")" == "$ENV_STAMP" ]]; then
  echo "Reusing stamped ARC env at $TMP_ENV_ROOT"
else
  PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" TMP_ENV_ROOT="$TMP_ENV_ROOT" ENV_STAMP="$ENV_STAMP" \
    bash "$PROJECT_ROOT/scripts/arc_setup_elf_env.sh"
fi
conda activate "$TMP_ENV_ROOT"

if [[ ! -f "$TRAIN_NEIGH" ]]; then
  python scripts/augment_sherlock_oracle_grammatical_neighborhood.py \
    --source-npz "$TRAIN" \
    --blocked-npz "$VAL" \
    --output "$TRAIN_NEIGH" \
    --variants-per-source 64 \
    --batch-size 512 \
    --device cuda
fi

if [[ ! -f "$EXTERNAL_NEIGH" ]]; then
  python scripts/augment_sherlock_oracle_grammatical_neighborhood.py \
    --source-npz "$SUITE_DIR/clean_balanced_selected400k_minilm_seed7.npz" \
    --source-limit 20000 \
    --blocked-npz "$TRAIN" \
    --blocked-npz "$VAL" \
    --output "$EXTERNAL_NEIGH" \
    --variants-per-source 32 \
    --batch-size 512 \
    --device cuda
fi

build_and_score() {
  local tag=$1
  shift
  local combined=$SUITE_DIR/${tag}_to_sherlock12_combined_minilm_seed7.npz
  python scripts/concat_semantic_npzs.py \
    "$@" \
    --val-npz "$VAL" \
    --output "$combined" \
    --dedupe-train \
    --dedupe-val \
    --drop-train-overlap-with-val
  python scripts/analyze_npz_coverage.py \
    --npz "$combined" \
    --name "$tag" \
    --val-num-examples 75 \
    --train-query-sample-size 1024 \
    --batch-size 64 \
    --seed 7 \
    --device cuda \
    --output-json "$SUITE_DIR/${tag}_coverage.json"
}

build_and_score clean_hybrid_200k \
  --train-npz "$TRAIN" --train-limit 0 \
  --train-npz "$SUITE_DIR/clean_balanced_selected400k_minilm_seed7.npz" --train-limit 50000 \
  --train-npz "$SUITE_DIR/clean_lexical_selected400k_minilm_seed7.npz" --train-limit 50000 \
  --train-npz "$SUITE_DIR/clean_rare_selected400k_minilm_seed7.npz" --train-limit 50000 \
  --train-npz "$SUITE_DIR/clean_semantic_selected400k_minilm_seed7.npz" --train-limit 50000

build_and_score clean_hybrid_400k \
  --train-npz "$TRAIN" --train-limit 0 \
  --train-npz "$SUITE_DIR/clean_balanced_selected400k_minilm_seed7.npz" --train-limit 100000 \
  --train-npz "$SUITE_DIR/clean_lexical_selected400k_minilm_seed7.npz" --train-limit 100000 \
  --train-npz "$SUITE_DIR/clean_rare_selected400k_minilm_seed7.npz" --train-limit 100000 \
  --train-npz "$SUITE_DIR/clean_semantic_selected400k_minilm_seed7.npz" --train-limit 100000

build_and_score clean_balanced200k_plus_trainneigh \
  --train-npz "$TRAIN" --train-limit 0 \
  --train-npz "$SUITE_DIR/clean_balanced_selected400k_minilm_seed7.npz" --train-limit 200000 \
  --train-npz "$TRAIN_NEIGH" --train-limit 150000

build_and_score clean_balanced50k_plus_externalneigh \
  --train-npz "$TRAIN" --train-limit 0 \
  --train-npz "$SUITE_DIR/clean_balanced_selected400k_minilm_seed7.npz" --train-limit 50000 \
  --train-npz "$EXTERNAL_NEIGH" --train-limit 300000

build_and_score clean_grand_hybrid400k \
  --train-npz "$TRAIN" --train-limit 0 \
  --train-npz "$SUITE_DIR/clean_balanced_selected400k_minilm_seed7.npz" --train-limit 75000 \
  --train-npz "$SUITE_DIR/clean_lexical_selected400k_minilm_seed7.npz" --train-limit 75000 \
  --train-npz "$SUITE_DIR/clean_rare_selected400k_minilm_seed7.npz" --train-limit 75000 \
  --train-npz "$SUITE_DIR/clean_semantic_selected400k_minilm_seed7.npz" --train-limit 75000 \
  --train-npz "$TRAIN_NEIGH" --train-limit 50000 \
  --train-npz "$EXTERNAL_NEIGH" --train-limit 50000
