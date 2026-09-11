#!/usr/bin/env bash
#SBATCH --job-name=qc4w_calib
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=submit-jobs/calibrate_qc4wyals_ada-%j.out
#SBATCH --error=submit-jobs/calibrate_qc4wyals_ada-%j.err

set -euo pipefail

project_root=/data/engs-pnpl/glandau/BrainDiffusion/ELF
interface_dir=/data/engs-pnpl/glandau/elf-cache/tang_themoth_qc4wyals_ada_v2/interface
output_dir=/data/engs-pnpl/glandau/elf-cache/tang_themoth_qc4wyals_ada_v2/calibration

cd "$project_root"
exec /data/engs-pnpl/glandau/conda/envs/meg2sem/bin/python \
  scripts/calibrate_qc4wyals_ada.py \
  --interface-dir "$interface_dir" \
  --output-dir "$output_dir"
