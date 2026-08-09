#!/usr/bin/env bash
#SBATCH --job-name=prep_sva_aug_v2
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=submit-jobs/prepare_libribrain_sva_train_aug_v2_strict_minilm_seed7-%j.out
#SBATCH --error=submit-jobs/prepare_libribrain_sva_train_aug_v2_strict_minilm_seed7-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}

RUN_TAG=${RUN_TAG:-libribrain_simple_sva_no_coord_decl5to18_sherlock1to9_mocha_timit_moth_valmix}
SVA_TRAIN=${SVA_TRAIN:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_train_minilm_seed7.npz}
SVA_VAL=${SVA_VAL:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_val_minilm_seed7.npz}
AUG3=${AUG3:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_trainaug_v2_strict3x_minilm_seed7.npz}
AUG5=${AUG5:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_trainaug_v2_strict5x_minilm_seed7.npz}
COMBINED3=${COMBINED3:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_plus_trainaug_v2_strict3x_combined_minilm_seed7.npz}
COMBINED5=${COMBINED5:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_plus_trainaug_v2_strict5x_combined_minilm_seed7.npz}
COVERAGE_JSON=${COVERAGE_JSON:-$DATA_ROOT/coverage/libribrain_sva_trainaug_v2_strict_coverage.json}

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

mkdir -p "$(dirname "$AUG3")" "$(dirname "$COVERAGE_JSON")" "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"
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

if [[ ! -f "$AUG3" ]]; then
  python scripts/augment_semantic_npz_sentences.py \
    --train-npz "$SVA_TRAIN" \
    --blocked-npz "$SVA_VAL" \
    --output "$AUG3" \
    --variants-per-sentence 3 \
    --quality-mode strict \
    --min-source-word-recall 0.75 \
    --max-source-word-recall 0.95 \
    --min-token-count 3 \
    --min-phrase-count 2 \
    --max-replacement-word-len 14 \
    --seed 7 \
    --batch-size 512 \
    --device cuda
fi

if [[ ! -f "$AUG5" ]]; then
  python scripts/augment_semantic_npz_sentences.py \
    --train-npz "$SVA_TRAIN" \
    --blocked-npz "$SVA_VAL" \
    --output "$AUG5" \
    --variants-per-sentence 5 \
    --quality-mode strict \
    --min-source-word-recall 0.75 \
    --max-source-word-recall 0.95 \
    --min-token-count 3 \
    --min-phrase-count 2 \
    --max-replacement-word-len 14 \
    --seed 7 \
    --batch-size 512 \
    --device cuda
fi

python scripts/concat_semantic_npzs.py \
  --train-npz "$SVA_TRAIN" \
  --train-npz "$AUG3" \
  --val-npz "$SVA_VAL" \
  --output "$COMBINED3" \
  --dedupe-train \
  --dedupe-val \
  --drop-train-overlap-with-val

python scripts/concat_semantic_npzs.py \
  --train-npz "$SVA_TRAIN" \
  --train-npz "$AUG5" \
  --val-npz "$SVA_VAL" \
  --output "$COMBINED5" \
  --dedupe-train \
  --dedupe-val \
  --drop-train-overlap-with-val

python scripts/analyze_npz_coverage.py \
  --npz "$DATA_ROOT/libribrain_simple/${RUN_TAG}_combined_minilm_seed7.npz" \
  --name base_sva \
  --npz "$DATA_ROOT/libribrain_simple/${RUN_TAG}_plus_trainaug_v1_5x_combined_minilm_seed7.npz" \
  --name trainaug_v1_5x \
  --npz "$COMBINED3" \
  --name trainaug_v2_strict3x \
  --npz "$COMBINED5" \
  --name trainaug_v2_strict5x \
  --val-num-examples 322 \
  --train-query-sample-size 4096 \
  --batch-size 128 \
  --seed 7 \
  --device cuda \
  --output-json "$COVERAGE_JSON"

echo "Prepared strict augmentation datasets:"
echo "$COMBINED3"
echo "$COMBINED5"
echo "Coverage report: $COVERAGE_JSON"
