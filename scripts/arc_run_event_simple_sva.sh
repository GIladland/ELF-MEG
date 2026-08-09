#!/usr/bin/env bash
set -euo pipefail

# Build and train a MiniLM-pooled ELF run from simple declarative LibriBrain event sentences.

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
LIBRIBRAIN100_ROOT=${LIBRIBRAIN100_ROOT:-/data/engs-pnpl/glandau/MEG2SEM/LibriBrain100}
LIBRIBRAIN2_ROOT=${LIBRIBRAIN2_ROOT:-/data/engs-pnpl/datasets/LibriBrain2}

FILTER_MODE=${FILTER_MODE:-simple_sv_no_coord}
FILTER_MIN_WORDS=${FILTER_MIN_WORDS:-5}
FILTER_MAX_WORDS=${FILTER_MAX_WORDS:-18}
FILTER_MIN_ALPHA_FRACTION=${FILTER_MIN_ALPHA_FRACTION:-0.75}
FILTER_MIN_CONTENT_WORDS=${FILTER_MIN_CONTENT_WORDS:-2}
VAL_INCLUDE_REGEX=${VAL_INCLUDE_REGEX:-'(/Sherlock1/.*_ses-(11|12)_|/TIMIT/.*_ses-14_|/MOCHATIMIT/.*_ses-4_|/TheMoth/.*_ses-29_)'}
MIN_VAL_EXAMPLES=${MIN_VAL_EXAMPLES:-200}

EMBEDDING_MODEL=${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}
EMBEDDING_POOLING=${EMBEDDING_POOLING:-mean}
EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-512}
EMBEDDING_MAX_LENGTH=${EMBEDDING_MAX_LENGTH:-128}

RUN_TAG=${RUN_TAG:-libribrain_simple_sva_no_coord_decl5to18_sherlock1to9_mocha_timit_moth_valmix}
TRAIN_OUTPUT=${TRAIN_OUTPUT:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_train_minilm_seed7.npz}
VAL_OUTPUT=${VAL_OUTPUT:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_val_minilm_seed7.npz}
COMBINED_OUTPUT=${COMBINED_OUTPUT:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_combined_minilm_seed7.npz}
OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/${RUN_TAG}_k128_epoch300_long2d}
TARGET_LATENTS_CACHE=${TARGET_LATENTS_CACHE:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_combined_t5small_latents_fp16.npy}

WANDB_RUN_NAME=${WANDB_RUN_NAME:-${RUN_TAG}_k128_epoch300_long2d}
WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-conditioning_minilm_simple_sva}
PREP_ONLY=${PREP_ONLY:-0}

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
  "$OUTPUT_DIR" \
  "$(dirname "$TRAIN_OUTPUT")" \
  "$(dirname "$VAL_OUTPUT")" \
  "$(dirname "$COMBINED_OUTPUT")" \
  "$(dirname "$TARGET_LATENTS_CACHE")" \
  "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"
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

EVENT_GLOB_ARGS=(
  --event-glob "$LIBRIBRAIN100_ROOT/Sherlock[1-7]/derivatives/events/*_events.tsv"
  --event-glob "$LIBRIBRAIN2_ROOT/Sherlock[89]/derivatives/events/*_events.tsv"
  --event-glob "$LIBRIBRAIN2_ROOT/TIMIT/derivatives/events/*_events.tsv"
  --event-glob "$LIBRIBRAIN2_ROOT/MOCHATIMIT/derivatives/events/*_events.tsv"
  --event-glob "$LIBRIBRAIN2_ROOT/TheMoth/derivatives/events/*_events.tsv"
)
EXPORT_COMMON_ARGS=(
  --filter-mode "$FILTER_MODE"
  --min-words "$FILTER_MIN_WORDS"
  --max-words "$FILTER_MAX_WORDS"
  --min-alpha-fraction "$FILTER_MIN_ALPHA_FRACTION"
  --min-content-words "$FILTER_MIN_CONTENT_WORDS"
  --dedupe
  --embedding-model-name "$EMBEDDING_MODEL"
  --pooling "$EMBEDDING_POOLING"
  --batch-size "$EMBEDDING_BATCH_SIZE"
  --max-length "$EMBEDDING_MAX_LENGTH"
  --device cuda
)

if [[ ! -f "$TRAIN_OUTPUT" ]]; then
  (
    flock -x 9
    if [[ ! -f "$TRAIN_OUTPUT" ]]; then
      python scripts/export_event_sentence_embeddings.py \
        --output "$TRAIN_OUTPUT" \
        "${EVENT_GLOB_ARGS[@]}" \
        --exclude-path-regex "$VAL_INCLUDE_REGEX" \
        "${EXPORT_COMMON_ARGS[@]}"
    fi
  ) 9>"$TRAIN_OUTPUT.lock"
fi

if [[ ! -f "$VAL_OUTPUT" ]]; then
  (
    flock -x 9
    if [[ ! -f "$VAL_OUTPUT" ]]; then
      python scripts/export_event_sentence_embeddings.py \
        --output "$VAL_OUTPUT" \
        "${EVENT_GLOB_ARGS[@]}" \
        --include-path-regex "$VAL_INCLUDE_REGEX" \
        "${EXPORT_COMMON_ARGS[@]}"
    fi
  ) 9>"$VAL_OUTPUT.lock"
fi

if [[ ! -f "$COMBINED_OUTPUT" ]]; then
  (
    flock -x 9
    if [[ ! -f "$COMBINED_OUTPUT" ]]; then
      python scripts/concat_semantic_npzs.py \
        --train-npz "$TRAIN_OUTPUT" \
        --train-limit 0 \
        --val-npz "$VAL_OUTPUT" \
        --output "$COMBINED_OUTPUT" \
        --dedupe-train \
        --dedupe-val \
        --drop-train-overlap-with-val
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
echo "Training simple SVA COMBINED_OUTPUT=$COMBINED_OUTPUT total=$TOTAL_N train=$TRAIN_N val=$VAL_N"
if (( VAL_N < MIN_VAL_EXAMPLES )); then
  echo "Validation set too small: val=$VAL_N < MIN_VAL_EXAMPLES=$MIN_VAL_EXAMPLES" >&2
  exit 3
fi

if [[ "$PREP_ONLY" == "1" ]]; then
  echo "PREP_ONLY=1: prepared COMBINED_OUTPUT=$COMBINED_OUTPUT and skipping ELF training."
  exit 0
fi

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
  --target-latents-mode cache \
  --target-latents-cache "$TARGET_LATENTS_CACHE" \
  --target-latents-cache-dtype float16 \
  "$@"
