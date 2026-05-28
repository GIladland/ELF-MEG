#!/bin/bash
# Minimal ARC smoke test for PyTorch ELF inference on a single GPU.
#
# Usage:
#   sbatch scripts/arc_eval_smoke.sbatch
#
# Expected setup on ARC before submission:
#   1. This repo is available on the cluster filesystem.
#   2. A Python/conda environment named `elf` exists with `pip install -r requirements.txt`.
#   3. The compute nodes can either reach Hugging Face, or you override the checkpoint/data
#      paths below to local cached copies.

#SBATCH --job-name=elf-smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -euo pipefail

# Override these at submit time if needed, e.g.
#   sbatch --export=ALL,PROJECT_ROOT=/path/to/ELF,OUTPUT_ROOT=/scratch/$USER/elf scripts/arc_eval_smoke.sbatch
PROJECT_ROOT=${PROJECT_ROOT:-$HOME/BrainDiffusion/ELF}
OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/$USER/elf}
CONDA_ENV=${CONDA_ENV:-elf}

# Hugging Face caches on scratch to avoid filling $HOME.
export HF_HOME=${HF_HOME:-$OUTPUT_ROOT/hf}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}

mkdir -p "$OUTPUT_ROOT" "$HF_HOME" "$PROJECT_ROOT/logs"

if command -v module >/dev/null 2>&1; then
    module purge || true
fi

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Could not find conda.sh; edit scripts/arc_eval_smoke.sbatch for your ARC environment."
    exit 1
fi

conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

echo "== Host =="
hostname
date
pwd

echo "== CUDA =="
nvidia-smi || true

echo "== Python =="
python --version

echo "== Smoke test =="
bash scripts/launch.sh eval src/configs/training_configs/train_owt_ELF-B.yml \
  --checkpoint_path embedded-language-flows/ELF-B-owt-torch \
  --config_override global_batch_size=8 \
  --config_override num_samples=32 \
  --config_override use_bf16=true \
  --config_override use_compile=false \
  --config_override use_wandb=false \
  --config_override output_dir="$OUTPUT_ROOT/elf_eval_smoke"
