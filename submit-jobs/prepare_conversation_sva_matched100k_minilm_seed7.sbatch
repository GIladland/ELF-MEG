#!/usr/bin/env bash
#SBATCH --job-name=prep_conv_sva100k
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=submit-jobs/prepare_conversation_sva_matched100k_minilm_seed7-%j.out
#SBATCH --error=submit-jobs/prepare_conversation_sva_matched100k_minilm_seed7-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}

CONV_DIR=${CONV_DIR:-$DATA_ROOT/conversation}
DAILY_OUT=${DAILY_OUT:-$CONV_DIR/daily_dialog_simple_sv_no_coord_minilm_seed7.npz}
DAILY_FALLBACK_OUT=${DAILY_FALLBACK_OUT:-$CONV_DIR/dailydialog_hf_simple_sv_no_coord_minilm_seed7.npz}
EMPATHETIC_OUT=${EMPATHETIC_OUT:-$CONV_DIR/empathetic_dialogues_simple_sv_no_coord_minilm_seed7.npz}
EMPATHETIC_FALLBACK_OUT=${EMPATHETIC_FALLBACK_OUT:-$CONV_DIR/empathetic_dialogues_legacy_simple_sv_no_coord_minilm_seed7.npz}
SAMSUM_OUT=${SAMSUM_OUT:-$CONV_DIR/samsum_simple_sv_no_coord_minilm_seed7.npz}
SAMSUM_FALLBACK_OUT=${SAMSUM_FALLBACK_OUT:-$CONV_DIR/samsum_legacy_simple_sv_no_coord_minilm_seed7.npz}
MERGED_CONV=${MERGED_CONV:-$CONV_DIR/conversation_clean_simple_sv_no_coord_candidates_minilm_seed7.npz}
SELECTED_CONV=${SELECTED_CONV:-$CONV_DIR/conversation_clean_simple_sv_no_coord_matched100k_by_libribrain_sva_train_minilm_seed7.npz}

RUN_TAG=${RUN_TAG:-libribrain_simple_sva_no_coord_decl5to18_sherlock1to9_mocha_timit_moth_valmix}
SVA_TRAIN=${SVA_TRAIN:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_train_minilm_seed7.npz}
SVA_VAL=${SVA_VAL:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_val_minilm_seed7.npz}
SVA_COMBINED=${SVA_COMBINED:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_combined_minilm_seed7.npz}
OUTPUT_COMBINED=${OUTPUT_COMBINED:-$DATA_ROOT/libribrain_simple/${RUN_TAG}_plus_conversation_svamatch100k_combined_minilm_seed7.npz}

SELECT_COUNT=${SELECT_COUNT:-100000}
QUERY_SAMPLE_SIZE=${QUERY_SAMPLE_SIZE:-0}
PER_QUERY_CANDIDATES=${PER_QUERY_CANDIDATES:-32}
CANDIDATE_BATCH_SIZE=${CANDIDATE_BATCH_SIZE:-4096}
LIMIT_SENTENCES_PER_SOURCE=${LIMIT_SENTENCES_PER_SOURCE:-150000}
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

mkdir -p "$CONV_DIR" "$(dirname "$OUTPUT_COMBINED")" "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"
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

COMMON_EXPORT_ARGS=(
  --filter-mode simple_sv_no_coord
  --min-words 5
  --max-words 18
  --min-alpha-fraction 0.75
  --min-content-words 2
  --dedupe
  --limit-sentences "$LIMIT_SENTENCES_PER_SOURCE"
  --embedding-model-name sentence-transformers/all-MiniLM-L6-v2
  --pooling mean
  --batch-size 512
  --max-length 128
  --device cuda
  --trust-remote-code
)

try_export() {
  local output=$1
  shift
  if [[ -f "$output" ]]; then
    echo "Using existing conversation source: $output"
    return 0
  fi
  echo "Exporting conversation source to $output"
  if python scripts/export_hf_conversation_sentence_embeddings.py --output "$output" "$@" "${COMMON_EXPORT_ARGS[@]}"; then
    return 0
  fi
  echo "WARNING: export failed for $output" >&2
  rm -f "$output" "${output%.npz}.summary.json"
  return 1
}

try_export "$DAILY_OUT" \
  --dataset daily_dialog \
  --split train \
  --text-field dialog || true

if [[ ! -f "$DAILY_OUT" ]]; then
  try_export "$DAILY_FALLBACK_OUT" \
    --dataset frankdarkluo/DailyDialog \
    --split train \
    --text-field dialog \
    --text-field dialogue \
    --text-field utterance \
    --text-field text \
    --infer-text-fields || true
fi

try_export "$EMPATHETIC_OUT" \
  --dataset facebook/empathetic_dialogues \
  --split train \
  --text-field utterance || true

if [[ ! -f "$EMPATHETIC_OUT" ]]; then
  try_export "$EMPATHETIC_FALLBACK_OUT" \
    --dataset empathetic_dialogues \
    --split train \
    --text-field utterance || true
fi

try_export "$SAMSUM_OUT" \
  --dataset knkarthick/samsum \
  --split train \
  --text-field dialogue || true

if [[ ! -f "$SAMSUM_OUT" ]]; then
  try_export "$SAMSUM_FALLBACK_OUT" \
    --dataset samsum \
    --split train \
    --text-field dialogue || true
fi

echo "Merging conversational candidate NPZs."
python scripts/merge_semantic_npzs.py \
  --input-npz "$DAILY_OUT" \
  --input-npz "$DAILY_FALLBACK_OUT" \
  --input-npz "$EMPATHETIC_OUT" \
  --input-npz "$EMPATHETIC_FALLBACK_OUT" \
  --input-npz "$SAMSUM_OUT" \
  --input-npz "$SAMSUM_FALLBACK_OUT" \
  --output "$MERGED_CONV" \
  --dedupe

echo "Selecting $SELECT_COUNT conversational SVA rows matched to LibriBrain/Sherlock SVA train."
python scripts/select_semantic_npz_by_similarity.py \
  --candidate-npz "$MERGED_CONV" \
  --query-npz "$SVA_TRAIN" \
  --output "$SELECTED_CONV" \
  --count "$SELECT_COUNT" \
  --selection-mode query_balanced \
  --query-sample-size "$QUERY_SAMPLE_SIZE" \
  --per-query-candidates "$PER_QUERY_CANDIDATES" \
  --candidate-batch-size "$CANDIDATE_BATCH_SIZE" \
  --blocked-npz "$SVA_COMBINED" \
  --seed "$SEED" \
  --device cuda

echo "Combining current SVA train + matched conversational 100k with held-out SVA validation."
python scripts/concat_semantic_npzs.py \
  --train-npz "$SVA_TRAIN" \
  --train-npz "$SELECTED_CONV" \
  --val-npz "$SVA_VAL" \
  --output "$OUTPUT_COMBINED" \
  --dedupe-train \
  --dedupe-val \
  --drop-train-overlap-with-val

echo "Prepared combined conversational dataset: $OUTPUT_COMBINED"
