#!/usr/bin/env bash
#SBATCH --job-name=sh_cov_select
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-18:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=220G
#SBATCH --output=submit-jobs/prepare_sherlock_sva_clean_select400k-%j.out
#SBATCH --error=submit-jobs/prepare_sherlock_sva_clean_select400k-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
STRATEGY=${STRATEGY:?Set STRATEGY to balanced, lexical, rare, or semantic}
SUITE_DIR=${SUITE_DIR:-$DATA_ROOT/sherlock_sva/coverage_suite_v1}

TRAIN=$DATA_ROOT/sherlock_sva/sherlock1to9_sva_no_coord_decl5to18_train_minilm_seed7.npz
VAL=$DATA_ROOT/sherlock_sva/sherlock12_sva_no_coord_decl5to18_val_minilm_seed7.npz
GUTENBERG=$DATA_ROOT/gutenberg/gutenberg_catalog_en_short5to18_simple_sv_no_coord_minilm_seed7.npz
CONVERSATION=$DATA_ROOT/conversation/conversation_clean_simple_sv_no_coord_candidates_minilm_seed7.npz
SELECTED=$SUITE_DIR/clean_${STRATEGY}_selected400k_minilm_seed7.npz

case "$STRATEGY" in
  balanced)
    COSINE_WEIGHT=0.35; LEXICAL_WEIGHT=0.40; LENGTH_WEIGHT=0.10; ORDER_WEIGHT=0.15; LEXICAL_MODE=idf_f1
    ;;
  lexical)
    COSINE_WEIGHT=0.15; LEXICAL_WEIGHT=0.55; LENGTH_WEIGHT=0.10; ORDER_WEIGHT=0.20; LEXICAL_MODE=word_f1
    ;;
  rare)
    COSINE_WEIGHT=0.20; LEXICAL_WEIGHT=0.60; LENGTH_WEIGHT=0.10; ORDER_WEIGHT=0.10; LEXICAL_MODE=idf_recall
    ;;
  semantic)
    COSINE_WEIGHT=0.70; LEXICAL_WEIGHT=0.20; LENGTH_WEIGHT=0.10; ORDER_WEIGHT=0.00; LEXICAL_MODE=idf_f1
    ;;
  *)
    echo "Unknown STRATEGY=$STRATEGY" >&2
    exit 2
    ;;
esac

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

python scripts/select_semantic_npz_by_coverage.py \
  --candidate-npz "$GUTENBERG" \
  --candidate-name gutenberg_clean_sva \
  --candidate-npz "$CONVERSATION" \
  --candidate-name conversation_sva \
  --query-npz "$TRAIN" \
  --blocked-npz "$TRAIN" \
  --blocked-npz "$VAL" \
  --exclude-candidate-row-regex 'Doyle, Arthur Conan|Conan Doyle|Sherlock' \
  --output "$SELECTED" \
  --count 400000 \
  --per-query-candidates 48 \
  --per-query-lexical-candidates 128 \
  --lexical-token-max-candidates 12000 \
  --candidate-batch-size 2048 \
  --cosine-weight "$COSINE_WEIGHT" \
  --lexical-weight "$LEXICAL_WEIGHT" \
  --length-weight "$LENGTH_WEIGHT" \
  --order-weight "$ORDER_WEIGHT" \
  --lexical-mode "$LEXICAL_MODE" \
  --max-per-query 64 \
  --seed 7 \
  --device cuda

for BUDGET in 50000 100000 200000 400000; do
  TAG=clean_${STRATEGY}_${BUDGET}
  COMBINED=$SUITE_DIR/${TAG}_to_sherlock12_combined_minilm_seed7.npz
  REPORT=$SUITE_DIR/${TAG}_coverage.json
  python scripts/concat_semantic_npzs.py \
    --train-npz "$TRAIN" \
    --train-limit 0 \
    --train-npz "$SELECTED" \
    --train-limit "$BUDGET" \
    --val-npz "$VAL" \
    --output "$COMBINED" \
    --dedupe-train \
    --dedupe-val \
    --drop-train-overlap-with-val
  python scripts/analyze_npz_coverage.py \
    --npz "$COMBINED" \
    --name "$TAG" \
    --val-num-examples 75 \
    --train-query-sample-size 1024 \
    --batch-size 64 \
    --seed 7 \
    --device cuda \
    --output-json "$REPORT"
done
