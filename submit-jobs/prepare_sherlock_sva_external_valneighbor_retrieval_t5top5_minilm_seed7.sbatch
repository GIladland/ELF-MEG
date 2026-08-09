#!/usr/bin/env bash
#SBATCH --job-name=sh_ext_retr
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-14:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=170G
#SBATCH --output=submit-jobs/prepare_sherlock_sva_external_valneighbor_retrieval_t5top5_minilm_seed7-%j.out
#SBATCH --error=submit-jobs/prepare_sherlock_sva_external_valneighbor_retrieval_t5top5_minilm_seed7-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
SUITE_DIR=${SUITE_DIR:-$DATA_ROOT/sherlock_sva/coverage_suite_v1}
SOURCE_NPZ=${SOURCE_NPZ:-$SUITE_DIR/external_valneighbor20k_to_sherlock12_combined_minilm_seed7.npz}
SELECTED_BANK=${SELECTED_BANK:-$SUITE_DIR/external_valneighbor_selected20k_minilm_seed7.npz}
TRAIN_BANK=${TRAIN_BANK:-$DATA_ROOT/sherlock_sva/sherlock1to9_sva_no_coord_decl5to18_train_minilm_seed7.npz}
RETRIEVAL_NPZ=${RETRIEVAL_NPZ:-$SUITE_DIR/external_valneighbor20k_retrievalt5top5_to_sherlock12_combined_minilm_seed7.npz}
RUN_NAME=${RUN_NAME:-sherlock_sva_external_valneighbor20k_retrievalt5top5_minilm_k64_epoch120_fullmask_long2d}
EPOCHS=${EPOCHS:-120}

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

for required in "$SOURCE_NPZ" "$SELECTED_BANK" "$TRAIN_BANK"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required NPZ: $required" >&2
    exit 1
  fi
done

if [[ ! -f "$RETRIEVAL_NPZ" ]]; then
  python scripts/build_retrieval_augmented_npz.py \
    --npz "$SOURCE_NPZ" \
    --output "$RETRIEVAL_NPZ" \
    --bank-npz "$SELECTED_BANK" \
    --bank-npz "$TRAIN_BANK" \
    --neighbor-top-k 5 \
    --retrieval-batch-size 128 \
    --t5-batch-size 128 \
    --encoder-model-name t5-small \
    --latent-mean 0.0 \
    --latent-std 0.2 \
    --device cuda \
    --exclude-identical-text \
    --include-scores
fi

export RUN_NAME
export NPZ_PATH="$RETRIEVAL_NPZ"
export EPOCHS
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-sherlock_target_neighbor_diagnostics}
export WANDB_NOTES_TEXT=${WANDB_NOTES_TEXT:-Diagnostic retrieval-context run. Uses Sherlock12 held-out MiniLM/text only to select nearby non-Sherlock external candidates, blocks exact held-out and Sherlock train text, then appends top-5 retrieved T5 pooled neighbor features to the MiniLM query. Not a clean generalization result; tests whether top-k lexical neighborhood information fixes pooled MiniLM ambiguity.}

sbatch submit-jobs/sherlock_sva_coverage_gate_train_minilm_k64.sbatch
