#!/usr/bin/env bash
#SBATCH --job-name=fmri_elf_test
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=submit-jobs/test_fmri_elf_content_winner-%j.out
#SBATCH --error=submit-jobs/test_fmri_elf_content_winner-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
environment_root=${TMP_ENV_ROOT:-$data_root/conda-envs/braindiffusion-elf-torch271cu118}
export PATH="$environment_root/bin:$PATH"
export PYTHONNOUSERSITE=1
cd "$project_root"

python tests/test_elf_brain_cross_attention.py
python tests/test_fmri2sem_bridge.py
python tests/test_generation_token_bias.py
