#!/usr/bin/env bash
#SBATCH --job-name=export_sva_meg
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=submit-jobs/export_sherlock_sva_meg2sem_minilm-%j.out
#SBATCH --error=submit-jobs/export_sherlock_sva_meg2sem_minilm-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
EXPORT_CONDA_ENV=${EXPORT_CONDA_ENV:-/data/engs-pnpl/glandau/BrainDiffusion/elf-env}
USE_NODE_LOCAL_ENV=${USE_NODE_LOCAL_ENV:-0}

SVA_NPZ=${SVA_NPZ:-$DATA_ROOT/sherlock_sva/sherlock1to9_sva_no_coord_decl5to18_train_minilm_seed7.npz}
OUTPUT_NPZ=${OUTPUT_NPZ:-$DATA_ROOT/sherlock_sva/meg2sem/sherlock1to9_sva_no_coord_decl5to18_train_minilm_meg2sem_segment3000_fp16.npz}
SIDECAR_ROOT=${SIDECAR_ROOT:-$DATA_ROOT/sherlock_sva/meg2sem/minilm_sidecars}
SEGMENT_MS=${SEGMENT_MS:-3000}
MEG_DTYPE=${MEG_DTYPE:-float16}
MAX_ROWS=${MAX_ROWS:-0}
FORCE=${FORCE:-0}
STANDARDIZE=${STANDARDIZE:-1}

LIBRIBRAIN100_ROOT=${LIBRIBRAIN100_ROOT:-/data/engs-pnpl/glandau/MEG2SEM/LibriBrain100}
LIBRIBRAIN2_ROOT=${LIBRIBRAIN2_ROOT:-/data/engs-pnpl/datasets/LibriBrain2}
LIBRIBRAIN_ROOT=${LIBRIBRAIN_ROOT:-/data/engs-pnpl/glandau/MEG2SEM/LibriBrain}

if ! command -v conda >/dev/null 2>&1; then
  for conda_sh in \
    "${CONDA_SH:-}" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/data/engs-pnpl/glandau/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -n "$conda_sh" && -f "$conda_sh" ]]; then
      source "$conda_sh"
      break
    fi
  done
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found; set CONDA_SH to the conda.sh path before launching." >&2
  exit 127
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_ROOT/xdg-cache}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${TORCH_HOME:-$DATA_ROOT/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/pip-cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$DATA_ROOT/conda-pkgs}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TMPDIR="${TMPDIR:-$DATA_ROOT/tmp}"

mkdir -p "$(dirname "$OUTPUT_NPZ")" "$SIDECAR_ROOT" "$TMPDIR"
cd "$PROJECT_ROOT"

ENV_STAMP_PATH="$TMP_ENV_ROOT/.elf_env_stamp"
if [[ "$USE_NODE_LOCAL_ENV" != "1" && -d "$EXPORT_CONDA_ENV/bin" ]]; then
  echo "Using persistent export env: $EXPORT_CONDA_ENV"
  conda activate "$EXPORT_CONDA_ENV"
elif [[ "${SKIP_ARC_SETUP:-1}" == "1" && -d "$TMP_ENV_ROOT/bin" && -f "$ENV_STAMP_PATH" && "$(cat "$ENV_STAMP_PATH")" == "$ENV_STAMP" ]]; then
  echo "Skipping ARC env setup; reusing TMP_ENV_ROOT=$TMP_ENV_ROOT ENV_STAMP=$ENV_STAMP"
  conda activate "$TMP_ENV_ROOT"
else
  echo "Preparing ARC env at $TMP_ENV_ROOT"
  PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" TMP_ENV_ROOT="$TMP_ENV_ROOT" ENV_STAMP="$ENV_STAMP" \
    bash "$PROJECT_ROOT/scripts/arc_setup_elf_env.sh"
  conda activate "$TMP_ENV_ROOT"
fi

ARGS=(
  --sva-npz "$SVA_NPZ"
  --output "$OUTPUT_NPZ"
  --sidecar-root "$SIDECAR_ROOT"
  --segment-ms "$SEGMENT_MS"
  --meg-dtype "$MEG_DTYPE"
  --fallback-data-root "$LIBRIBRAIN100_ROOT"
  --fallback-data-root "$LIBRIBRAIN2_ROOT"
  --fallback-data-root "$LIBRIBRAIN_ROOT"
  --max-rows "$MAX_ROWS"
)

if [[ "$FORCE" == "1" ]]; then
  ARGS+=(--force)
fi
if [[ "$STANDARDIZE" != "1" ]]; then
  ARGS+=(--no-standardize)
fi

echo "Exporting Sherlock SVA MEG2SEM MiniLM dataset"
echo "SVA_NPZ=$SVA_NPZ"
echo "OUTPUT_NPZ=$OUTPUT_NPZ"
echo "SIDECAR_ROOT=$SIDECAR_ROOT"
echo "STANDARDIZE=$STANDARDIZE"
python scripts/export_sherlock_sva_meg2sem_npz.py "${ARGS[@]}"
