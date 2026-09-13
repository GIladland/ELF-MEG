#!/usr/bin/env bash
set -euo pipefail

# ARC GPU-node environment setup for ELF.
# Keeps the executable conda env on node-local /tmp, but sends all large caches
# and model/dataset downloads to /data instead of /home.

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}

# Several Slurm jobs can share a node, while their Conda package cache lives on
# shared /data. Serialize setup cluster-wide so jobs on different GPU nodes do
# not concurrently mutate that shared cache.
if command -v flock >/dev/null 2>&1 && [[ "${ARC_ENV_LOCK_HELD:-0}" != "1" ]]; then
  mkdir -p "$DATA_ROOT/locks"
  lock_path=$DATA_ROOT/locks/${USER}-braindiffusion-elf-env-setup.lock
  exec flock --wait 1800 "$lock_path" env ARC_ENV_LOCK_HELD=1 bash "$0" "$@"
fi

# Multiple jobs can fail the caller-side stamp check before the first setup
# finishes, then wait on the lock above. Re-check after acquiring the lock so
# those waiters reuse the completed node-local environment instead of
# reinstalling it serially.
stamp_path=$TMP_ENV_ROOT/.elf_env_stamp
if [[ "${ARC_ENV_LOCK_HELD:-0}" == "1" && -d "$TMP_ENV_ROOT/bin" && -f "$stamp_path" ]]; then
  if [[ "$(cat "$stamp_path")" == "$ENV_STAMP" ]]; then
    echo "Environment became ready while waiting for setup lock; reusing TMP_ENV_ROOT=$TMP_ENV_ROOT"
    exit 0
  fi
fi

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
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$DATA_ROOT/torch}"
export WANDB_DIR="${WANDB_DIR:-$DATA_ROOT/wandb}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$DATA_ROOT/triton-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/pip-cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$DATA_ROOT/conda-pkgs}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

mkdir -p \
  "$PROJECT_ROOT" \
  "$DATA_ROOT" \
  "$XDG_CACHE_HOME" \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TORCH_HOME" \
  "$WANDB_DIR" \
  "$TRITON_CACHE_DIR" \
  "$PIP_CACHE_DIR" \
  "$CONDA_PKGS_DIRS"

if [ ! -d "$TMP_ENV_ROOT/bin" ]; then
  conda create -y -p "$TMP_ENV_ROOT" python=3.10
fi

conda activate "$TMP_ENV_ROOT"
cd "$PROJECT_ROOT"

echo "HOST: $(hostname)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "TMP_ENV_ROOT=$TMP_ENV_ROOT"
echo "ENV_STAMP=$ENV_STAMP"
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "TORCH_HOME=$TORCH_HOME"
echo "WANDB_DIR=$WANDB_DIR"
echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
echo "PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "CONDA_PKGS_DIRS=$CONDA_PKGS_DIRS"

python --version
nvidia-smi || true

if [[ -n "${TORCH_PIP_SPEC:-}" ]]; then
  if [[ -z "${TORCH_PIP_INDEX_URL:-}" ]]; then
    echo "TORCH_PIP_INDEX_URL is required when TORCH_PIP_SPEC is set." >&2
    exit 2
  fi
  echo "Installing GPU-compatible torch override: $TORCH_PIP_SPEC"
  python -m pip install --no-user \
    --index-url "$TORCH_PIP_INDEX_URL" \
    "$TORCH_PIP_SPEC"
  # requirements.txt pins the default cluster torch build. The override above
  # intentionally replaces only that line while retaining every other project
  # dependency for older GPU generations such as V100 (sm_70).
  python -m pip install --no-user -r <(grep -vE '^torch([<=>!~[:space:]]|$)' requirements.txt)
else
  python -m pip install --no-user -r requirements.txt
fi
printf "%s\n" "$ENV_STAMP" > "$TMP_ENV_ROOT/.elf_env_stamp"

python - <<'PY'
import os
import torch, transformers, datasets, wandb
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_arch_list", torch.cuda.get_arch_list() if torch.cuda.is_available() else [])
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("wandb", wandb.__version__)
print("cuda_available", torch.cuda.is_available())
for key in [
    "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE", "TORCH_HOME", "WANDB_DIR",
    "PIP_CACHE_DIR", "CONDA_PKGS_DIRS", "XDG_CACHE_HOME", "TRITON_CACHE_DIR",
    "PYTHONNOUSERSITE",
]:
    print(key, os.environ.get(key))
PY
