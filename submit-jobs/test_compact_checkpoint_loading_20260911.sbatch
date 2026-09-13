#!/usr/bin/env bash
#SBATCH --job-name=elfcompacttest
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --oversubscribe
#SBATCH --output=submit-jobs/test_compact_checkpoint_loading_20260911-%j.out
#SBATCH --error=submit-jobs/test_compact_checkpoint_loading_20260911-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
python_bin=/data/engs-pnpl/glandau/elf-cache/conda-envs/braindiffusion-elf-torch271cu118/bin/python
cd "$project_root"
exec "$python_bin" -u tests/test_compact_checkpoint_loading.py
