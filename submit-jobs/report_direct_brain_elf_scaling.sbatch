#!/usr/bin/env bash
#SBATCH --job-name=elf_direct_scale_report
#SBATCH --clusters=arc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=submit-jobs/report_direct_brain_elf_scaling-%j.out
#SBATCH --error=submit-jobs/report_direct_brain_elf_scaling-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
python_bin=/data/engs-pnpl/glandau/elf-cache/conda-envs/braindiffusion-elf-torch271cu118/bin/python

"$python_bin" "$project_root/scripts/report_direct_brain_elf_scaling.py" \
  --output-markdown "$project_root/ELF_DIRECT_BRAIN_TO_SCALED_DIFFUSION_2026-09-03.md" \
  --output-json "$project_root/ELF_DIRECT_BRAIN_TO_SCALED_DIFFUSION_2026-09-03.json"
