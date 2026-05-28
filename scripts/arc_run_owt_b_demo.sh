#!/usr/bin/env bash
set -euo pipefail

# ARC GPU-node ELF-B OWT demo run.
# Uses node-local /tmp for the executable env, and /data for all large caches.

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-elf-env}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/engs-pnpl/glandau/elf-runs/owt-b-demo}

source "$(conda info --base)/etc/profile.d/conda.sh"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_ROOT/xdg-cache}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$DATA_ROOT/torch}"
export WANDB_DIR="${WANDB_DIR:-$DATA_ROOT/wandb}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/pip-cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$DATA_ROOT/conda-pkgs}"

mkdir -p "$OUTPUT_ROOT"

cd "$PROJECT_ROOT"

if [ ! -d "$TMP_ENV_ROOT/bin" ]; then
  PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" TMP_ENV_ROOT="$TMP_ENV_ROOT" \
    "$PROJECT_ROOT/scripts/arc_setup_elf_env.sh"
fi

conda activate "$TMP_ENV_ROOT"

echo "HOST: $(hostname)"
python --version
nvidia-smi || true

bash scripts/launch.sh eval src/configs/training_configs/train_owt_ELF-B.yml \
  --checkpoint_path embedded-language-flows/ELF-B-owt-torch \
  --config_override global_batch_size=8 \
  --config_override num_samples=16 \
  --config_override use_bf16=true \
  --config_override use_compile=false \
  --config_override use_wandb=false \
  --config_override output_dir="$OUTPUT_ROOT"

LATEST_JSONL="$(find "$OUTPUT_ROOT" -type f -name 'all_generated_*.jsonl' | sort | tail -n 1)"
echo "LATEST_JSONL=$LATEST_JSONL"

python - <<'PY' "$LATEST_JSONL"
import json, sys
path = sys.argv[1]
print("SAMPLES_FROM", path)
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 8:
            break
        obj = json.loads(line)
        print(f"[{i}] {obj['generated']}")
PY
