#!/usr/bin/env bash
#SBATCH --job-name=setup_elf_shared
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=submit-jobs/setup_arc_shared_elf_env-%j.out
#SBATCH --error=submit-jobs/setup_arc_shared_elf_env-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export TORCH_PIP_SPEC=${TORCH_PIP_SPEC:-torch==2.7.1}
export TORCH_PIP_INDEX_URL=${TORCH_PIP_INDEX_URL:-https://download.pytorch.org/whl/cu118}

exec bash "$PROJECT_ROOT/scripts/arc_setup_elf_env.sh"
