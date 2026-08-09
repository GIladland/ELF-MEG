#!/usr/bin/env bash
#SBATCH --job-name=sh_gpt2_cov
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-14:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=submit-jobs/prepare_sherlock_sva_gpt2large_coverage-%j.out
#SBATCH --error=submit-jobs/prepare_sherlock_sva_gpt2large_coverage-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
SUITE_DIR=${SUITE_DIR:-$DATA_ROOT/sherlock_sva/coverage_suite_v1}
TRAIN=$DATA_ROOT/sherlock_sva/sherlock1to9_sva_no_coord_decl5to18_train_minilm_seed7.npz
VAL=$DATA_ROOT/sherlock_sva/sherlock12_sva_no_coord_decl5to18_val_minilm_seed7.npz
LM_TAG=${LM_TAG:-gpt2large_aug}
LM_BANK=${LM_BANK:-$SUITE_DIR/sherlock1to9_gpt2large_aug200k_minilm_seed7.npz}
LM_COLA=${LM_COLA:-$SUITE_DIR/${LM_TAG}_cola_minilm_seed7.npz}
LM_SELECTED=${LM_SELECTED:-$SUITE_DIR/${LM_TAG}_selected_minilm_seed7.npz}
LM_COLA_SELECTED=${LM_COLA_SELECTED:-$SUITE_DIR/${LM_TAG}_cola_selected_minilm_seed7.npz}
SOURCE_INDEX_KEY=${SOURCE_INDEX_KEY:-prompt_source_index}

source "$(conda info --base)/etc/profile.d/conda.sh"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$DATA_ROOT/xdg-cache}
export HF_HOME=${HF_HOME:-$DATA_ROOT/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
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

python scripts/filter_semantic_npz_by_grammar.py \
  --input "$LM_BANK" \
  --output "$LM_COLA" \
  --source-index-key "$SOURCE_INDEX_KEY" \
  --min-grammar-score 0.55 \
  --min-source-content-recall 0.0 \
  --max-per-source 64 \
  --batch-size 256 \
  --device cuda \
  --local-files-only

select_bank() {
  local bank=$1
  local output=$2
  python scripts/select_semantic_npz_by_coverage.py \
    --candidate-npz "$bank" \
    --candidate-name gpt2large_sherlock_train_aug \
    --query-npz "$TRAIN" \
    --blocked-npz "$TRAIN" \
    --blocked-npz "$VAL" \
    --output "$output" \
    --count 200000 \
    --per-query-candidates 64 \
    --per-query-lexical-candidates 128 \
    --lexical-token-max-candidates 12000 \
    --candidate-batch-size 2048 \
    --cosine-weight 0.35 \
    --lexical-weight 0.40 \
    --length-weight 0.10 \
    --order-weight 0.15 \
    --lexical-mode idf_f1 \
    --max-per-query 64 \
    --seed 7 \
    --device cuda
}

select_bank "$LM_BANK" "$LM_SELECTED"
select_bank "$LM_COLA" "$LM_COLA_SELECTED"

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

for BUDGET in 50000 100000 200000; do
  build_and_score ${LM_TAG}_${BUDGET} \
    --train-npz "$TRAIN" --train-limit 0 \
    --train-npz "$LM_SELECTED" --train-limit "$BUDGET"
  build_and_score ${LM_TAG}_cola_${BUDGET} \
    --train-npz "$TRAIN" --train-limit 0 \
    --train-npz "$LM_COLA_SELECTED" --train-limit "$BUDGET"
done

build_and_score ${LM_TAG}100k_plus_cleanbalanced100k \
  --train-npz "$TRAIN" --train-limit 0 \
  --train-npz "$LM_SELECTED" --train-limit 100000 \
  --train-npz "$SUITE_DIR/clean_balanced_selected400k_minilm_seed7.npz" --train-limit 100000
