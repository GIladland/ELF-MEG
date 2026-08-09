#!/usr/bin/env bash
#SBATCH --job-name=prep_sh_aug10
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=submit-jobs/prepare_sherlock_sva_train_aug10_minilm_seed7-%j.out
#SBATCH --error=submit-jobs/prepare_sherlock_sva_train_aug10_minilm_seed7-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}

MIXED_TAG=${MIXED_TAG:-libribrain_simple_sva_no_coord_decl5to18_sherlock1to9_mocha_timit_moth_valmix}
MIXED_TRAIN=${MIXED_TRAIN:-$DATA_ROOT/libribrain_simple/${MIXED_TAG}_train_minilm_seed7.npz}
MIXED_VAL=${MIXED_VAL:-$DATA_ROOT/libribrain_simple/${MIXED_TAG}_val_minilm_seed7.npz}

SHERLOCK_DIR=${SHERLOCK_DIR:-$DATA_ROOT/sherlock_sva}
SHERLOCK_TRAIN=${SHERLOCK_TRAIN:-$SHERLOCK_DIR/sherlock1to9_sva_no_coord_decl5to18_train_minilm_seed7.npz}
SHERLOCK_VAL=${SHERLOCK_VAL:-$SHERLOCK_DIR/sherlock12_sva_no_coord_decl5to18_val_minilm_seed7.npz}

BASE_COMBINED=${BASE_COMBINED:-$SHERLOCK_DIR/sherlock1to9_sva_no_coord_decl5to18_to_sherlock12_sva_no_coord_decl5to18_combined_minilm_seed7.npz}
AUG5_COMBINED=${AUG5_COMBINED:-$SHERLOCK_DIR/sherlock1to9_sva_no_coord_decl5to18_plus_trainaug_v1_5x_to_sherlock12_sva_no_coord_decl5to18_combined_minilm_seed7.npz}
AUG10_TAG=${AUG10_TAG:-sherlock1to9_sva_no_coord_decl5to18_plus_trainaug_v1_10x_to_sherlock12_sva_no_coord_decl5to18}
AUG10=${AUG10:-$SHERLOCK_DIR/sherlock1to9_sva_no_coord_decl5to18_trainaug_v1_10x_minilm_seed7.npz}
AUG10_COMBINED=${AUG10_COMBINED:-$SHERLOCK_DIR/${AUG10_TAG}_combined_minilm_seed7.npz}
COVERAGE_JSON=${COVERAGE_JSON:-$DATA_ROOT/coverage/sherlock_sva_trainaug10_coverage.json}

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

mkdir -p "$SHERLOCK_DIR" "$(dirname "$COVERAGE_JSON")" "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"
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

if [[ ! -f "$SHERLOCK_TRAIN" ]]; then
  python scripts/filter_semantic_npz_rows.py \
    --input "$MIXED_TRAIN" \
    --output "$SHERLOCK_TRAIN" \
    --include-field-regex "book=^Sherlock" \
    --include-field-regex "task=^Sherlock"
fi

if [[ ! -f "$SHERLOCK_VAL" ]]; then
  python scripts/filter_semantic_npz_rows.py \
    --input "$MIXED_VAL" \
    --output "$SHERLOCK_VAL" \
    --include-field-regex "book=^Sherlock" \
    --include-field-regex "task=^Sherlock"
fi

if [[ ! -f "$AUG10" ]]; then
  python scripts/augment_semantic_npz_sentences.py \
    --train-npz "$SHERLOCK_TRAIN" \
    --blocked-npz "$SHERLOCK_VAL" \
    --output "$AUG10" \
    --variants-per-sentence 10 \
    --quality-mode loose \
    --seed 17 \
    --batch-size 512 \
    --device cuda
fi

python scripts/concat_semantic_npzs.py \
  --train-npz "$SHERLOCK_TRAIN" \
  --train-npz "$AUG10" \
  --val-npz "$SHERLOCK_VAL" \
  --output "$AUG10_COMBINED" \
  --dedupe-train \
  --dedupe-val \
  --drop-train-overlap-with-val

VAL_N=$(python - "$SHERLOCK_VAL" <<'PY'
import sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=True) as data:
    print(len(data["sentence"]))
PY
)

NPZ_ARGS=(--npz "$AUG10_COMBINED" --name sherlock_sva_trainaug_v1_10x)
if [[ -f "$BASE_COMBINED" ]]; then
  NPZ_ARGS=(--npz "$BASE_COMBINED" --name sherlock_sva_base "${NPZ_ARGS[@]}")
fi
if [[ -f "$AUG5_COMBINED" ]]; then
  NPZ_ARGS=(--npz "$AUG5_COMBINED" --name sherlock_sva_trainaug_v1_5x "${NPZ_ARGS[@]}")
fi

python scripts/analyze_npz_coverage.py \
  "${NPZ_ARGS[@]}" \
  --val-num-examples "$VAL_N" \
  --train-query-sample-size 4096 \
  --batch-size 128 \
  --seed 7 \
  --device cuda \
  --output-json "$COVERAGE_JSON"

echo "Prepared Sherlock-only SVA 10x augmentation dataset:"
echo "$AUG10_COMBINED"
echo "Coverage report: $COVERAGE_JSON"
