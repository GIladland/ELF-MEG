#!/usr/bin/env bash
#SBATCH --job-name=elfm_intprep
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=submit-jobs/prepare_qc4wyals_elfm_interpolation_ladder_20260906-%j.out
#SBATCH --error=submit-jobs/prepare_qc4wyals_elfm_interpolation_ladder_20260906-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
interface=$data_root/tang_themoth_qc4wyals_ada_v2/interface
python=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
exec "$python" "$project_root/scripts/build_semantic_interpolation_ladder.py" \
  --predicted "$interface/qc4wyals_val_predicted_ada002.npz" \
  --exact "$interface/qc4wyals_val_exact_ada002.npz" \
  --output-dir "$interface/interpolation_val110_20260906"
