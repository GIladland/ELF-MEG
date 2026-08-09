#!/usr/bin/env bash
set -euo pipefail

# Build a train-first / Sherlock1-session-11-12-validation MiniLM NPZ, then train ELF.

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
LIBRIBRAIN_DATA_ROOT=${LIBRIBRAIN_DATA_ROOT:-/data/engs-pnpl/glandau/MEG2SEM/LibriBrain}
OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/minilm-sherlock-holdout}
COMBINED_OUTPUT=${COMBINED_OUTPUT:-$DATA_ROOT/sherlock/minilm_sherlock_holdout_combined.npz}

SYNTHETIC_OUTPUT=${SYNTHETIC_OUTPUT:-}
SYNTHETIC_COUNT=${SYNTHETIC_COUNT:-0}
SYNTHETIC_LIMIT=${SYNTHETIC_LIMIT:-0}
SHERLOCK_TRAIN_OUTPUT=${SHERLOCK_TRAIN_OUTPUT:-$DATA_ROOT/sherlock/sherlock_books1to9_sentences_minilm_seed7.npz}
SHERLOCK_VAL_OUTPUT=${SHERLOCK_VAL_OUTPUT:-$DATA_ROOT/sherlock/sherlock1_ses11_12_sentences_minilm_seed7.npz}
INCLUDE_SHERLOCK_TRAIN=${INCLUDE_SHERLOCK_TRAIN:-0}
SHERLOCK_TRAIN_BOOKS=${SHERLOCK_TRAIN_BOOKS:-"1 2 3 4 5 6 7 8 9"}
SHERLOCK_VAL_SESSIONS=${SHERLOCK_VAL_SESSIONS:-"11 12"}

EMBEDDING_MODEL=${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}
EMBEDDING_POOLING=${EMBEDDING_POOLING:-mean}
EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-512}
EMBEDDING_MAX_LENGTH=${EMBEDDING_MAX_LENGTH:-128}

WANDB_RUN_NAME=${WANDB_RUN_NAME:-minilm-sherlock-holdout}
WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-conditioning_minilm_sherlock_holdout}

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

mkdir -p "$OUTPUT_DIR" "$(dirname "$COMBINED_OUTPUT")" "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"
cd "$PROJECT_ROOT"

ENV_STAMP_PATH="$TMP_ENV_ROOT/.elf_env_stamp"
if [[ "${SKIP_ARC_SETUP:-0}" == "1" && -d "$TMP_ENV_ROOT/bin" && -f "$ENV_STAMP_PATH" && "$(cat "$ENV_STAMP_PATH")" == "$ENV_STAMP" ]]; then
  echo "Skipping ARC env setup; reusing TMP_ENV_ROOT=$TMP_ENV_ROOT ENV_STAMP=$ENV_STAMP"
elif [[ "${SKIP_ARC_SETUP:-0}" == "1" ]]; then
  echo "SKIP_ARC_SETUP=1 requested, but matching env stamp was not found; falling back to setup: $TMP_ENV_ROOT"
  PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" TMP_ENV_ROOT="$TMP_ENV_ROOT" ENV_STAMP="$ENV_STAMP" \
    bash "$PROJECT_ROOT/scripts/arc_setup_elf_env.sh"
else
  PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" TMP_ENV_ROOT="$TMP_ENV_ROOT" ENV_STAMP="$ENV_STAMP" \
    bash "$PROJECT_ROOT/scripts/arc_setup_elf_env.sh"
fi

conda activate "$TMP_ENV_ROOT"

if [[ -n "$SYNTHETIC_OUTPUT" && "$SYNTHETIC_COUNT" != "0" && ! -f "$SYNTHETIC_OUTPUT" ]]; then
  mkdir -p "$(dirname "$SYNTHETIC_OUTPUT")"
  (
    flock -x 9
    if [[ ! -f "$SYNTHETIC_OUTPUT" ]]; then
      python scripts/export_synthetic_sentence_embeddings.py \
        --output "$SYNTHETIC_OUTPUT" \
        --count "$SYNTHETIC_COUNT" \
        --seed 7 \
        --embedding-model-name "$EMBEDDING_MODEL" \
        --pooling "$EMBEDDING_POOLING" \
        --batch-size "$EMBEDDING_BATCH_SIZE" \
        --max-length "$EMBEDDING_MAX_LENGTH" \
        --device cuda
    fi
  ) 9>"$SYNTHETIC_OUTPUT.lock"
fi

mkdir -p "$(dirname "$SHERLOCK_VAL_OUTPUT")"
if [[ ! -f "$SHERLOCK_VAL_OUTPUT" ]]; then
  (
    flock -x 9
    if [[ ! -f "$SHERLOCK_VAL_OUTPUT" ]]; then
      read -r -a val_sessions <<< "$SHERLOCK_VAL_SESSIONS"
      python scripts/export_libribrain_sentence_embeddings.py \
        --data-root "$LIBRIBRAIN_DATA_ROOT" \
        --output "$SHERLOCK_VAL_OUTPUT" \
        --sherlock1-sessions "${val_sessions[@]}" \
        --dedupe \
        --embedding-model-name "$EMBEDDING_MODEL" \
        --pooling "$EMBEDDING_POOLING" \
        --batch-size "$EMBEDDING_BATCH_SIZE" \
        --max-length "$EMBEDDING_MAX_LENGTH" \
        --device cuda
    fi
  ) 9>"$SHERLOCK_VAL_OUTPUT.lock"
fi

if [[ "$INCLUDE_SHERLOCK_TRAIN" == "1" && ! -f "$SHERLOCK_TRAIN_OUTPUT" ]]; then
  mkdir -p "$(dirname "$SHERLOCK_TRAIN_OUTPUT")"
  (
    flock -x 9
    if [[ ! -f "$SHERLOCK_TRAIN_OUTPUT" ]]; then
      read -r -a train_books <<< "$SHERLOCK_TRAIN_BOOKS"
      python scripts/export_libribrain_sentence_embeddings.py \
        --data-root "$LIBRIBRAIN_DATA_ROOT" \
        --output "$SHERLOCK_TRAIN_OUTPUT" \
        --books "${train_books[@]}" \
        --dedupe \
        --embedding-model-name "$EMBEDDING_MODEL" \
        --pooling "$EMBEDDING_POOLING" \
        --batch-size "$EMBEDDING_BATCH_SIZE" \
        --max-length "$EMBEDDING_MAX_LENGTH" \
        --device cuda
    fi
  ) 9>"$SHERLOCK_TRAIN_OUTPUT.lock"
fi

if [[ ! -f "$COMBINED_OUTPUT" ]]; then
  (
    flock -x 9
    if [[ ! -f "$COMBINED_OUTPUT" ]]; then
      concat_args=(--val-npz "$SHERLOCK_VAL_OUTPUT" --output "$COMBINED_OUTPUT" --dedupe-train --dedupe-val --drop-train-overlap-with-val)
      if [[ -n "$SYNTHETIC_OUTPUT" ]]; then
        concat_args+=(--train-npz "$SYNTHETIC_OUTPUT" --train-limit "$SYNTHETIC_LIMIT")
      fi
      if [[ "$INCLUDE_SHERLOCK_TRAIN" == "1" ]]; then
        concat_args+=(--train-npz "$SHERLOCK_TRAIN_OUTPUT" --train-limit 0)
      fi
      python scripts/concat_semantic_npzs.py "${concat_args[@]}"
    fi
  ) 9>"$COMBINED_OUTPUT.lock"
fi

read -r TOTAL_N VAL_N TRAIN_N < <(
  python - "$COMBINED_OUTPUT" <<'PY'
import json
import sys
import numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
summary = json.loads(str(z["schema_json"].tolist()))
print(summary["total_examples"], summary["val_examples"], summary["train_examples"])
PY
)
echo "Training on COMBINED_OUTPUT=$COMBINED_OUTPUT total=$TOTAL_N train=$TRAIN_N val=$VAL_N"

python scripts/train_npz_semantic_to_elf.py \
  --npz-path "$COMBINED_OUTPUT" \
  --use_wandb \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_group "$WANDB_RUN_GROUP" \
  --wandb_run_name "$WANDB_RUN_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --device cuda \
  --num-examples "$TOTAL_N" \
  --val-num-examples "$VAL_N" \
  --eval-num-examples "$VAL_N" \
  --retrieval-num-examples "$VAL_N" \
  "$@"
