#!/usr/bin/env bash
#SBATCH --job-name=elf_arch_report
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=submit-jobs/report_elf_architecture_scaling-%j.out
#SBATCH --error=submit-jobs/report_elf_architecture_scaling-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
python "$project_root/scripts/report_elf_architecture_scaling.py" \
  --output-json "$project_root/ELF_ARCHITECTURE_SCALING_X1_2026-09-04.json" \
  --output-markdown "$project_root/ELF_ARCHITECTURE_SCALING_X1_2026-09-04.md"
