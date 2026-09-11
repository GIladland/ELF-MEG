#!/usr/bin/env bash
#SBATCH --job-name=val_temporal_res
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=submit-jobs/validate_fmri_temporal_residual-%j.out
#SBATCH --error=submit-jobs/validate_fmri_temporal_residual-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=${PYTHON_BIN:-$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python}
cd "$project_root"
export PYTHONPATH="$project_root:$project_root/src:$project_root/scripts:${PYTHONPATH:-}"
"$python_bin" tests/test_fmri_temporal_residual.py
"$python_bin" tests/test_fmri_t5_delay_calibrator.py
"$python_bin" tests/test_fmri_t5_prefix_smoke.py
