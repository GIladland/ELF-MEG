#!/usr/bin/env bash
#SBATCH --job-name=prep_sh_emb5x
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=submit-jobs/prepare_sherlock_sva_embedding_compare_5x_seed7-%j.out
#SBATCH --error=submit-jobs/prepare_sherlock_sva_embedding_compare_5x_seed7-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}

SHERLOCK_DIR=${SHERLOCK_DIR:-$DATA_ROOT/sherlock_sva}
BASE_TAG=${BASE_TAG:-sherlock1to9_sva_no_coord_decl5to18_plus_trainaug_v1_5x_to_sherlock12_sva_no_coord_decl5to18}

TRAIN_MINILM=${TRAIN_MINILM:-$SHERLOCK_DIR/sherlock1to9_sva_no_coord_decl5to18_train_minilm_seed7.npz}
VAL_MINILM=${VAL_MINILM:-$SHERLOCK_DIR/sherlock12_sva_no_coord_decl5to18_val_minilm_seed7.npz}
AUG5_MINILM=${AUG5_MINILM:-$SHERLOCK_DIR/sherlock1to9_sva_no_coord_decl5to18_trainaug_v1_5x_minilm_seed7.npz}
COMBINED_MINILM=${COMBINED_MINILM:-$SHERLOCK_DIR/${BASE_TAG}_combined_minilm_seed7.npz}

TRAIN_MPNET=${TRAIN_MPNET:-$SHERLOCK_DIR/sherlock1to9_sva_no_coord_decl5to18_train_mpnet_seed7.npz}
VAL_MPNET=${VAL_MPNET:-$SHERLOCK_DIR/sherlock12_sva_no_coord_decl5to18_val_mpnet_seed7.npz}
AUG5_MPNET=${AUG5_MPNET:-$SHERLOCK_DIR/sherlock1to9_sva_no_coord_decl5to18_trainaug_v1_5x_mpnet_seed7.npz}
COMBINED_MPNET=${COMBINED_MPNET:-$SHERLOCK_DIR/${BASE_TAG}_combined_mpnet_seed7.npz}

TRAIN_BGE=${TRAIN_BGE:-$SHERLOCK_DIR/sherlock1to9_sva_no_coord_decl5to18_train_bgelargecls_seed7.npz}
VAL_BGE=${VAL_BGE:-$SHERLOCK_DIR/sherlock12_sva_no_coord_decl5to18_val_bgelargecls_seed7.npz}
AUG5_BGE=${AUG5_BGE:-$SHERLOCK_DIR/sherlock1to9_sva_no_coord_decl5to18_trainaug_v1_5x_bgelargecls_seed7.npz}
COMBINED_BGE=${COMBINED_BGE:-$SHERLOCK_DIR/${BASE_TAG}_combined_bgelargecls_seed7.npz}
COVERAGE_JSON=${COVERAGE_JSON:-$DATA_ROOT/coverage/sherlock_sva_embedding_compare_5x_coverage.json}

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

for required in "$TRAIN_MINILM" "$VAL_MINILM" "$AUG5_MINILM"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required MiniLM source NPZ: $required" >&2
    exit 1
  fi
done

if [[ ! -f "$TRAIN_MPNET" ]]; then
  python scripts/reembed_semantic_npz_sentences.py \
    --input-npz "$TRAIN_MINILM" \
    --output "$TRAIN_MPNET" \
    --embedding-model-name sentence-transformers/all-mpnet-base-v2 \
    --pooling mean \
    --batch-size 128 \
    --device cuda
fi
if [[ ! -f "$VAL_MPNET" ]]; then
  python scripts/reembed_semantic_npz_sentences.py \
    --input-npz "$VAL_MINILM" \
    --output "$VAL_MPNET" \
    --embedding-model-name sentence-transformers/all-mpnet-base-v2 \
    --pooling mean \
    --batch-size 128 \
    --device cuda
fi
if [[ ! -f "$AUG5_MPNET" ]]; then
  python scripts/reembed_semantic_npz_sentences.py \
    --input-npz "$AUG5_MINILM" \
    --output "$AUG5_MPNET" \
    --embedding-model-name sentence-transformers/all-mpnet-base-v2 \
    --pooling mean \
    --batch-size 128 \
    --device cuda
fi

if [[ ! -f "$TRAIN_BGE" ]]; then
  python scripts/reembed_semantic_npz_sentences.py \
    --input-npz "$TRAIN_MINILM" \
    --output "$TRAIN_BGE" \
    --embedding-model-name BAAI/bge-large-en-v1.5 \
    --pooling cls \
    --batch-size 64 \
    --device cuda
fi
if [[ ! -f "$VAL_BGE" ]]; then
  python scripts/reembed_semantic_npz_sentences.py \
    --input-npz "$VAL_MINILM" \
    --output "$VAL_BGE" \
    --embedding-model-name BAAI/bge-large-en-v1.5 \
    --pooling cls \
    --batch-size 64 \
    --device cuda
fi
if [[ ! -f "$AUG5_BGE" ]]; then
  python scripts/reembed_semantic_npz_sentences.py \
    --input-npz "$AUG5_MINILM" \
    --output "$AUG5_BGE" \
    --embedding-model-name BAAI/bge-large-en-v1.5 \
    --pooling cls \
    --batch-size 64 \
    --device cuda
fi

python scripts/concat_semantic_npzs.py \
  --train-npz "$TRAIN_MPNET" \
  --train-npz "$AUG5_MPNET" \
  --val-npz "$VAL_MPNET" \
  --output "$COMBINED_MPNET" \
  --dedupe-train \
  --dedupe-val \
  --drop-train-overlap-with-val

python scripts/concat_semantic_npzs.py \
  --train-npz "$TRAIN_BGE" \
  --train-npz "$AUG5_BGE" \
  --val-npz "$VAL_BGE" \
  --output "$COMBINED_BGE" \
  --dedupe-train \
  --dedupe-val \
  --drop-train-overlap-with-val

VAL_N=$(python - "$VAL_MINILM" <<'PY'
import sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=True) as data:
    print(len(data["sentence"]))
PY
)

NPZ_ARGS=()
if [[ -f "$COMBINED_MINILM" ]]; then
  NPZ_ARGS+=(--npz "$COMBINED_MINILM" --name minilm_5x)
fi
NPZ_ARGS+=(--npz "$COMBINED_MPNET" --name mpnet_5x)
NPZ_ARGS+=(--npz "$COMBINED_BGE" --name bgelargecls_5x)

python scripts/analyze_npz_coverage.py \
  "${NPZ_ARGS[@]}" \
  --val-num-examples "$VAL_N" \
  --train-query-sample-size 4096 \
  --batch-size 128 \
  --seed 7 \
  --device cuda \
  --output-json "$COVERAGE_JSON"

echo "Prepared embedding comparison datasets:"
echo "$COMBINED_MPNET"
echo "$COMBINED_BGE"
echo "Coverage report: $COVERAGE_JSON"
