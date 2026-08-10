#!/usr/bin/env bash
#SBATCH --job-name=prep_sh_ada
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=submit-jobs/prepare_sherlock_sva_oracle_valaug_ada_seed7-%j.out
#SBATCH --error=submit-jobs/prepare_sherlock_sva_oracle_valaug_ada_seed7-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
SHERLOCK_DIR=${SHERLOCK_DIR:-$DATA_ROOT/sherlock_sva}
ORACLE_TAG=${ORACLE_TAG:-sherlock1to9_sva_no_coord_decl5to18_plus_oraclevalaug64x_to_sherlock12_sva_no_coord_decl5to18}
SOURCE_NPZ=${SOURCE_NPZ:-$SHERLOCK_DIR/${ORACLE_TAG}_combined_minilm_seed7.npz}
ADA_NPZ=${ADA_NPZ:-$SHERLOCK_DIR/${ORACLE_TAG}_combined_ada002_seed7.npz}
ADA_MODEL=${ADA_MODEL:-text-embedding-ada-002}
ADA_BATCH_SIZE=${ADA_BATCH_SIZE:-128}
PYTHON_BIN=${PYTHON_BIN:-/data/engs-pnpl/glandau/BrainDiffusion/elf-env/bin/python}

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
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TMPDIR="${TMPDIR:-$DATA_ROOT/tmp}"

mkdir -p "$SHERLOCK_DIR" "$TMPDIR"
cd "$PROJECT_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 127
fi
if [[ ! -f "$SOURCE_NPZ" ]]; then
  echo "Missing source MiniLM oracle NPZ: $SOURCE_NPZ" >&2
  exit 1
fi
if [[ ! -f "$ADA_NPZ" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Missing OPENAI_API_KEY; needed to create $ADA_NPZ" >&2
  exit 2
fi

echo "Preparing true ADA oraclevalaug NPZ"
echo "SOURCE_NPZ=$SOURCE_NPZ"
echo "ADA_NPZ=$ADA_NPZ"
echo "ADA_MODEL=$ADA_MODEL"
echo "PYTHON_BIN=$PYTHON_BIN"

"$PYTHON_BIN" scripts/reembed_semantic_npz_openai.py \
  --input-npz "$SOURCE_NPZ" \
  --output "$ADA_NPZ" \
  --model "$ADA_MODEL" \
  --batch-size "$ADA_BATCH_SIZE"

"$PYTHON_BIN" - "$ADA_NPZ" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

path = Path(sys.argv[1])
summary = json.loads(path.with_suffix(".summary.json").read_text(encoding="utf-8"))
with np.load(path, allow_pickle=True) as data:
    embeddings = data["input_embeddings"]
    split = data["split"] if "split" in data.files else None
    print(
        json.dumps(
            {
                "path": str(path),
                "shape": list(embeddings.shape),
                "dtype": str(embeddings.dtype),
                "norm_mean": float(np.linalg.norm(embeddings[: min(1024, len(embeddings))], axis=1).mean()),
                "train_examples": summary.get("train_examples"),
                "val_examples": summary.get("val_examples"),
                "has_split": split is not None,
            },
            indent=2,
        )
    )
PY

echo "Prepared ADA NPZ: $ADA_NPZ"
