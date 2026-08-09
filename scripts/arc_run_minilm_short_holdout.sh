#!/usr/bin/env bash
set -euo pipefail

# Train MiniLM-pooled ELF on content-bearing short sentences, validated on Sherlock1 sessions 11/12.

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
LIBRIBRAIN_DATA_ROOT=${LIBRIBRAIN_DATA_ROOT:-/data/engs-pnpl/glandau/MEG2SEM/LibriBrain}

MODE=${MODE:-sherlock}  # sherlock | gutenberg | gutenberg_plus_sherlock
OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/minilm-short-holdout}
COMBINED_OUTPUT=${COMBINED_OUTPUT:-$DATA_ROOT/sherlock/minilm_short_holdout_combined.npz}

SHERLOCK_TRAIN_FULL_OUTPUT=${SHERLOCK_TRAIN_FULL_OUTPUT:-$DATA_ROOT/sherlock/sherlock_books1to9_sentences_minilm_seed7.npz}
SHERLOCK_VAL_FULL_OUTPUT=${SHERLOCK_VAL_FULL_OUTPUT:-$DATA_ROOT/sherlock/sherlock1_ses11_12_sentences_minilm_seed7.npz}
SHERLOCK_TRAIN_SHORT_OUTPUT=${SHERLOCK_TRAIN_SHORT_OUTPUT:-$DATA_ROOT/sherlock/sherlock_books1to9_short5to18_content_minilm_seed7.npz}
SHERLOCK_VAL_SHORT_OUTPUT=${SHERLOCK_VAL_SHORT_OUTPUT:-$DATA_ROOT/sherlock/sherlock1_ses11_12_short5to18_content_minilm_seed7.npz}
SHERLOCK_TRAIN_BOOKS=${SHERLOCK_TRAIN_BOOKS:-"1 2 3 4 5 6 7 8 9"}
SHERLOCK_VAL_SESSIONS=${SHERLOCK_VAL_SESSIONS:-"11 12"}

GUTENBERG_CACHE_DIR=${GUTENBERG_CACHE_DIR:-$DATA_ROOT/gutenberg/cache}
GUTENBERG_CANDIDATE_FULL_OUTPUT=${GUTENBERG_CANDIDATE_FULL_OUTPUT:-$DATA_ROOT/gutenberg/gutenberg_detective_nosherlock_candidates300k_books100_minilm_seed7.npz}
GUTENBERG_CANDIDATE_SHORT_OUTPUT=${GUTENBERG_CANDIDATE_SHORT_OUTPUT:-$DATA_ROOT/gutenberg/gutenberg_detective_nosherlock_candidates300k_books100_short5to18_content_minilm_seed7.npz}
GUTENBERG_SELECTED_OUTPUT=${GUTENBERG_SELECTED_OUTPUT:-$DATA_ROOT/gutenberg/gutenberg_detective_nosherlock_short5to18_selected100k_by_sherlock1to9_short5to18_minilm_seed7.npz}
GUTENBERG_CATALOG_CSV_URL=${GUTENBERG_CATALOG_CSV_URL:-}
GUTENBERG_SHUFFLE_BOOKS=${GUTENBERG_SHUFFLE_BOOKS:-0}
GUTENBERG_SELECTED_COUNT=${GUTENBERG_SELECTED_COUNT:-100000}
GUTENBERG_MAX_BOOKS=${GUTENBERG_MAX_BOOKS:-100}
GUTENBERG_LIMIT_SENTENCES=${GUTENBERG_LIMIT_SENTENCES:-300000}
GUTENBERG_QUERY_SAMPLE_SIZE=${GUTENBERG_QUERY_SAMPLE_SIZE:-8192}
GUTENBERG_SCORE_MODE=${GUTENBERG_SCORE_MODE:-max}
GUTENBERG_SELECTION_MODE=${GUTENBERG_SELECTION_MODE:-global}
GUTENBERG_PER_QUERY_CANDIDATES=${GUTENBERG_PER_QUERY_CANDIDATES:-64}
GUTENBERG_EXCLUDE_AUTHOR_REGEX=${GUTENBERG_EXCLUDE_AUTHOR_REGEX:-Arthur.*Conan.*Doyle|Doyle,.*Arthur.*Conan|Conan.*Doyle}
GUTENBERG_EXCLUDE_TITLE_REGEX=${GUTENBERG_EXCLUDE_TITLE_REGEX:-Sherlock|Holmes|Conan|Doyle|A Study in Scarlet|The Sign of the Four|Hound of the Baskervilles|Valley of Fear|Lock and Key Library|Library of the World}
GUTENBERG_EXCLUDE_SENTENCE_REGEX=${GUTENBERG_EXCLUDE_SENTENCE_REGEX:-Sherlock|Holmes|Watson|Conan Doyle|Briony Lodge|Irene Adler|Miss Sutherland|Baker Street}
GUTENBERG_INCLUDE_LOCC_REGEX=${GUTENBERG_INCLUDE_LOCC_REGEX:-}
GUTENBERG_EXCLUDE_LOCC_REGEX=${GUTENBERG_EXCLUDE_LOCC_REGEX:-}
GUTENBERG_INCLUDE_SUBJECT_REGEX=${GUTENBERG_INCLUDE_SUBJECT_REGEX:-}
GUTENBERG_EXCLUDE_SUBJECT_REGEX=${GUTENBERG_EXCLUDE_SUBJECT_REGEX:-}
GUTENBERG_INCLUDE_BOOKSHELF_REGEX=${GUTENBERG_INCLUDE_BOOKSHELF_REGEX:-}
GUTENBERG_EXCLUDE_BOOKSHELF_REGEX=${GUTENBERG_EXCLUDE_BOOKSHELF_REGEX:-}
GUTENBERG_REQUEST_TIMEOUT=${GUTENBERG_REQUEST_TIMEOUT:-90}
GUTENBERG_RETRIES=${GUTENBERG_RETRIES:-2}
GUTENBERG_DOWNLOAD_DELAY=${GUTENBERG_DOWNLOAD_DELAY:-0.05}

FILTER_MIN_WORDS=${FILTER_MIN_WORDS:-5}
FILTER_MAX_WORDS=${FILTER_MAX_WORDS:-18}
FILTER_MIN_ALPHA_FRACTION=${FILTER_MIN_ALPHA_FRACTION:-0.75}
FILTER_MIN_CONTENT_WORDS=${FILTER_MIN_CONTENT_WORDS:-2}

EMBEDDING_MODEL=${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}
EMBEDDING_POOLING=${EMBEDDING_POOLING:-mean}
EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-512}
EMBEDDING_MAX_LENGTH=${EMBEDDING_MAX_LENGTH:-128}

WANDB_RUN_NAME=${WANDB_RUN_NAME:-minilm-short-holdout}
WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-conditioning_minilm_short_holdout}
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
  "$(dirname "$COMBINED_OUTPUT")" \
  "$(dirname "$SHERLOCK_TRAIN_FULL_OUTPUT")" \
  "$(dirname "$SHERLOCK_TRAIN_SHORT_OUTPUT")" \
  "$(dirname "$GUTENBERG_CANDIDATE_FULL_OUTPUT")" \
  "$(dirname "$GUTENBERG_SELECTED_OUTPUT")" \
  "$GUTENBERG_CACHE_DIR" \
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

if [[ ! -f "$SHERLOCK_VAL_FULL_OUTPUT" ]]; then
  (
    flock -x 9
    if [[ ! -f "$SHERLOCK_VAL_FULL_OUTPUT" ]]; then
      read -r -a val_sessions <<< "$SHERLOCK_VAL_SESSIONS"
      python scripts/export_libribrain_sentence_embeddings.py \
        --data-root "$LIBRIBRAIN_DATA_ROOT" \
        --output "$SHERLOCK_VAL_FULL_OUTPUT" \
        --sherlock1-sessions "${val_sessions[@]}" \
        --dedupe \
        --embedding-model-name "$EMBEDDING_MODEL" \
        --pooling "$EMBEDDING_POOLING" \
        --batch-size "$EMBEDDING_BATCH_SIZE" \
        --max-length "$EMBEDDING_MAX_LENGTH" \
        --device cuda
    fi
  ) 9>"$SHERLOCK_VAL_FULL_OUTPUT.lock"
fi

if [[ ! -f "$SHERLOCK_TRAIN_FULL_OUTPUT" ]]; then
  (
    flock -x 9
    if [[ ! -f "$SHERLOCK_TRAIN_FULL_OUTPUT" ]]; then
      read -r -a train_books <<< "$SHERLOCK_TRAIN_BOOKS"
      python scripts/export_libribrain_sentence_embeddings.py \
        --data-root "$LIBRIBRAIN_DATA_ROOT" \
        --output "$SHERLOCK_TRAIN_FULL_OUTPUT" \
        --books "${train_books[@]}" \
        --dedupe \
        --embedding-model-name "$EMBEDDING_MODEL" \
        --pooling "$EMBEDDING_POOLING" \
        --batch-size "$EMBEDDING_BATCH_SIZE" \
        --max-length "$EMBEDDING_MAX_LENGTH" \
        --device cuda
    fi
  ) 9>"$SHERLOCK_TRAIN_FULL_OUTPUT.lock"
fi

if [[ ! -f "$SHERLOCK_TRAIN_SHORT_OUTPUT" ]]; then
  (
    flock -x 9
    if [[ ! -f "$SHERLOCK_TRAIN_SHORT_OUTPUT" ]]; then
      python scripts/filter_semantic_npz_sentences.py \
        --input "$SHERLOCK_TRAIN_FULL_OUTPUT" \
        --output "$SHERLOCK_TRAIN_SHORT_OUTPUT" \
        --min-words "$FILTER_MIN_WORDS" \
        --max-words "$FILTER_MAX_WORDS" \
        --min-alpha-fraction "$FILTER_MIN_ALPHA_FRACTION" \
        --min-content-words "$FILTER_MIN_CONTENT_WORDS" \
        --dedupe
    fi
  ) 9>"$SHERLOCK_TRAIN_SHORT_OUTPUT.lock"
fi

if [[ ! -f "$SHERLOCK_VAL_SHORT_OUTPUT" ]]; then
  (
    flock -x 9
    if [[ ! -f "$SHERLOCK_VAL_SHORT_OUTPUT" ]]; then
      python scripts/filter_semantic_npz_sentences.py \
        --input "$SHERLOCK_VAL_FULL_OUTPUT" \
        --output "$SHERLOCK_VAL_SHORT_OUTPUT" \
        --min-words "$FILTER_MIN_WORDS" \
        --max-words "$FILTER_MAX_WORDS" \
        --min-alpha-fraction "$FILTER_MIN_ALPHA_FRACTION" \
        --min-content-words "$FILTER_MIN_CONTENT_WORDS" \
        --dedupe
    fi
  ) 9>"$SHERLOCK_VAL_SHORT_OUTPUT.lock"
fi

case "$MODE" in
  sherlock)
    TRAIN_NPZS=("$SHERLOCK_TRAIN_SHORT_OUTPUT")
    ;;
  gutenberg|gutenberg_plus_sherlock)
    if [[ ! -f "$GUTENBERG_CANDIDATE_FULL_OUTPUT" ]]; then
      (
        flock -x 9
        if [[ ! -f "$GUTENBERG_CANDIDATE_FULL_OUTPUT" ]]; then
          gutenberg_args=(
            --output "$GUTENBERG_CANDIDATE_FULL_OUTPUT"
            --cache-dir "$GUTENBERG_CACHE_DIR"
            --dedupe
            --exclude-author-regex "$GUTENBERG_EXCLUDE_AUTHOR_REGEX"
            --exclude-title-regex "$GUTENBERG_EXCLUDE_TITLE_REGEX"
            --exclude-sentence-regex "$GUTENBERG_EXCLUDE_SENTENCE_REGEX"
            --include-locc-regex "$GUTENBERG_INCLUDE_LOCC_REGEX"
            --exclude-locc-regex "$GUTENBERG_EXCLUDE_LOCC_REGEX"
            --include-subject-regex "$GUTENBERG_INCLUDE_SUBJECT_REGEX"
            --exclude-subject-regex "$GUTENBERG_EXCLUDE_SUBJECT_REGEX"
            --include-bookshelf-regex "$GUTENBERG_INCLUDE_BOOKSHELF_REGEX"
            --exclude-bookshelf-regex "$GUTENBERG_EXCLUDE_BOOKSHELF_REGEX"
            --min-words "$FILTER_MIN_WORDS"
            --max-words "$FILTER_MAX_WORDS"
            --min-alpha-fraction "$FILTER_MIN_ALPHA_FRACTION"
            --request-timeout "$GUTENBERG_REQUEST_TIMEOUT"
            --retries "$GUTENBERG_RETRIES"
            --download-delay "$GUTENBERG_DOWNLOAD_DELAY"
            --embedding-model-name "$EMBEDDING_MODEL"
            --pooling "$EMBEDDING_POOLING"
            --batch-size "$EMBEDDING_BATCH_SIZE"
            --max-length "$EMBEDDING_MAX_LENGTH"
            --device cuda
          )
          if [[ -n "$GUTENBERG_CATALOG_CSV_URL" ]]; then
            gutenberg_args+=(--catalog-csv-url "$GUTENBERG_CATALOG_CSV_URL")
          fi
          if [[ "$GUTENBERG_SHUFFLE_BOOKS" == "1" ]]; then
            gutenberg_args+=(--shuffle-books --seed 7)
          fi
          if [[ "$GUTENBERG_MAX_BOOKS" != "0" ]]; then
            gutenberg_args+=(--max-books "$GUTENBERG_MAX_BOOKS")
          fi
          if [[ "$GUTENBERG_LIMIT_SENTENCES" != "0" ]]; then
            gutenberg_args+=(--limit-sentences "$GUTENBERG_LIMIT_SENTENCES")
          fi
          python scripts/export_gutenberg_sentence_embeddings.py "${gutenberg_args[@]}"
        fi
      ) 9>"$GUTENBERG_CANDIDATE_FULL_OUTPUT.lock"
    fi
    if [[ ! -f "$GUTENBERG_CANDIDATE_SHORT_OUTPUT" ]]; then
      (
        flock -x 9
        if [[ ! -f "$GUTENBERG_CANDIDATE_SHORT_OUTPUT" ]]; then
          python scripts/filter_semantic_npz_sentences.py \
            --input "$GUTENBERG_CANDIDATE_FULL_OUTPUT" \
            --output "$GUTENBERG_CANDIDATE_SHORT_OUTPUT" \
            --min-words "$FILTER_MIN_WORDS" \
            --max-words "$FILTER_MAX_WORDS" \
            --min-alpha-fraction "$FILTER_MIN_ALPHA_FRACTION" \
            --min-content-words "$FILTER_MIN_CONTENT_WORDS" \
            --dedupe
        fi
      ) 9>"$GUTENBERG_CANDIDATE_SHORT_OUTPUT.lock"
    fi
    if [[ ! -f "$GUTENBERG_SELECTED_OUTPUT" ]]; then
      (
        flock -x 9
        if [[ ! -f "$GUTENBERG_SELECTED_OUTPUT" ]]; then
          python scripts/select_semantic_npz_by_similarity.py \
            --candidate-npz "$GUTENBERG_CANDIDATE_SHORT_OUTPUT" \
            --query-npz "$SHERLOCK_TRAIN_SHORT_OUTPUT" \
            --blocked-npz "$SHERLOCK_TRAIN_SHORT_OUTPUT" \
            --blocked-npz "$SHERLOCK_VAL_SHORT_OUTPUT" \
            --output "$GUTENBERG_SELECTED_OUTPUT" \
            --count "$GUTENBERG_SELECTED_COUNT" \
            --query-sample-size "$GUTENBERG_QUERY_SAMPLE_SIZE" \
            --selection-mode "$GUTENBERG_SELECTION_MODE" \
            --per-query-candidates "$GUTENBERG_PER_QUERY_CANDIDATES" \
            --score-mode "$GUTENBERG_SCORE_MODE" \
            --device cuda \
            --seed 7
        fi
      ) 9>"$GUTENBERG_SELECTED_OUTPUT.lock"
    fi
    if [[ "$MODE" == "gutenberg_plus_sherlock" ]]; then
      TRAIN_NPZS=("$GUTENBERG_SELECTED_OUTPUT" "$SHERLOCK_TRAIN_SHORT_OUTPUT")
    else
      TRAIN_NPZS=("$GUTENBERG_SELECTED_OUTPUT")
    fi
    ;;
  *)
    echo "Unsupported MODE=$MODE; expected sherlock, gutenberg, or gutenberg_plus_sherlock" >&2
    exit 2
    ;;
esac

if [[ ! -f "$COMBINED_OUTPUT" ]]; then
  (
    flock -x 9
    if [[ ! -f "$COMBINED_OUTPUT" ]]; then
      concat_args=(
        --val-npz "$SHERLOCK_VAL_SHORT_OUTPUT"
        --output "$COMBINED_OUTPUT"
        --dedupe-train
        --dedupe-val
        --drop-train-overlap-with-val
      )
      for train_npz in "${TRAIN_NPZS[@]}"; do
        concat_args+=(--train-npz "$train_npz" --train-limit 0)
      done
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
echo "Training on MODE=$MODE COMBINED_OUTPUT=$COMBINED_OUTPUT total=$TOTAL_N train=$TRAIN_N val=$VAL_N"

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
  "$@"
