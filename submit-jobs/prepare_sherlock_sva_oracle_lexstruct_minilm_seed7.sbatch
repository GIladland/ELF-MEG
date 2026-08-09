#!/usr/bin/env bash
#SBATCH --job-name=prep_sh_lexstr
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=submit-jobs/prepare_sherlock_sva_oracle_lexstruct_minilm_seed7-%j.out
#SBATCH --error=submit-jobs/prepare_sherlock_sva_oracle_lexstruct_minilm_seed7-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}

SHERLOCK_DIR=${SHERLOCK_DIR:-$DATA_ROOT/sherlock_sva}
TRAIN=${TRAIN:-$SHERLOCK_DIR/sherlock1to9_sva_no_coord_decl5to18_train_minilm_seed7.npz}
VAL=${VAL:-$SHERLOCK_DIR/sherlock12_sva_no_coord_decl5to18_val_minilm_seed7.npz}

BASE_TAG=${BASE_TAG:-sherlock1to9_sva_no_coord_decl5to18_to_sherlock12_sva_no_coord_decl5to18}
ORACLE_TAG=${ORACLE_TAG:-sherlock1to9_sva_no_coord_decl5to18_plus_oraclevalaug64x_to_sherlock12_sva_no_coord_decl5to18}
STRUCT_TAG=${STRUCT_TAG:-sherlock1to9_sva_no_coord_decl5to18_plus_oraclestructure64x_to_sherlock12_sva_no_coord_decl5to18}
LEX_TAG=${LEX_TAG:-sherlock1to9_sva_no_coord_decl5to18_plus_oraclelexicontemplate64x_to_sherlock12_sva_no_coord_decl5to18}

BASE_COMBINED=${BASE_COMBINED:-$SHERLOCK_DIR/${BASE_TAG}_combined_minilm_seed7.npz}
ORACLE_COMBINED=${ORACLE_COMBINED:-$SHERLOCK_DIR/${ORACLE_TAG}_combined_minilm_seed7.npz}
STRUCT_AUG=${STRUCT_AUG:-$SHERLOCK_DIR/sherlock12_sva_no_coord_decl5to18_oracle_structure64x_minilm_seed7.npz}
LEX_AUG=${LEX_AUG:-$SHERLOCK_DIR/sherlock12_sva_no_coord_decl5to18_oracle_lexicontemplate64x_minilm_seed7.npz}
STRUCT_COMBINED=${STRUCT_COMBINED:-$SHERLOCK_DIR/${STRUCT_TAG}_combined_minilm_seed7.npz}
LEX_COMBINED=${LEX_COMBINED:-$SHERLOCK_DIR/${LEX_TAG}_combined_minilm_seed7.npz}
COVERAGE_JSON=${COVERAGE_JSON:-$DATA_ROOT/coverage/sherlock_sva_oracle_lexstruct64x_coverage.json}

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

for required in "$TRAIN" "$VAL" "$BASE_COMBINED"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required NPZ: $required" >&2
    exit 1
  fi
done

if [[ ! -f "$STRUCT_AUG" ]]; then
  python scripts/augment_sherlock_oracle_lexical_structure_ablation.py \
    --source-npz "$VAL" \
    --support-npz "$TRAIN" \
    --blocked-npz "$TRAIN" \
    --blocked-npz "$VAL" \
    --output "$STRUCT_AUG" \
    --mode structure_only \
    --variants-per-source 64 \
    --max-attempts-per-source 4000 \
    --filter-mode short_content \
    --max-structure-content-f1 0.35 \
    --seed 83 \
    --batch-size 512 \
    --device cuda
fi

if [[ ! -f "$LEX_AUG" ]]; then
  python scripts/augment_sherlock_oracle_lexical_structure_ablation.py \
    --source-npz "$VAL" \
    --support-npz "$TRAIN" \
    --blocked-npz "$TRAIN" \
    --blocked-npz "$VAL" \
    --output "$LEX_AUG" \
    --mode lexicon_template \
    --variants-per-source 64 \
    --max-attempts-per-source 4000 \
    --filter-mode short_content \
    --min-lexicon-content-recall 0.65 \
    --seed 89 \
    --batch-size 512 \
    --device cuda
fi

python scripts/concat_semantic_npzs.py \
  --train-npz "$TRAIN" \
  --train-npz "$STRUCT_AUG" \
  --val-npz "$VAL" \
  --output "$STRUCT_COMBINED" \
  --dedupe-train \
  --dedupe-val \
  --drop-train-overlap-with-val

python scripts/concat_semantic_npzs.py \
  --train-npz "$TRAIN" \
  --train-npz "$LEX_AUG" \
  --val-npz "$VAL" \
  --output "$LEX_COMBINED" \
  --dedupe-train \
  --dedupe-val \
  --drop-train-overlap-with-val

VAL_N=$(python - "$VAL" <<'PY'
import sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=True) as data:
    print(len(data["sentence"]))
PY
)

NPZ_ARGS=(
  --npz "$BASE_COMBINED" --name sherlock_sva_base
  --npz "$STRUCT_COMBINED" --name sherlock_sva_oraclestructure64x
  --npz "$LEX_COMBINED" --name sherlock_sva_oraclelexicontemplate64x
)
if [[ -f "$ORACLE_COMBINED" ]]; then
  NPZ_ARGS+=(--npz "$ORACLE_COMBINED" --name sherlock_sva_oracle_mixed64x)
fi

python scripts/analyze_npz_coverage.py \
  "${NPZ_ARGS[@]}" \
  --val-num-examples "$VAL_N" \
  --train-query-sample-size 4096 \
  --batch-size 128 \
  --seed 7 \
  --device cuda \
  --output-json "$COVERAGE_JSON"

echo "Prepared oracle lexicon/structure ablation datasets:"
echo "$STRUCT_AUG"
echo "$LEX_AUG"
echo "$STRUCT_COMBINED"
echo "$LEX_COMBINED"
echo "Coverage report: $COVERAGE_JSON"
