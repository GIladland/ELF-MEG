#!/usr/bin/env bash
#SBATCH --job-name=elf_scale_report
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=submit-jobs/report_elf_semantic_scaling_ladder-%j.out
#SBATCH --error=submit-jobs/report_elf_semantic_scaling_ladder-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
python_bin=${PYTHON_BIN:-/data/engs-pnpl/glandau/elf-cache/conda-envs/braindiffusion-elf-torch271cu118/bin/python}
"$python_bin" "$project_root/scripts/report_semantic_scaling_ladder.py" \
  --output-json "$project_root/ELF_SEMANTIC_SCALING_LADDER_2026-09-02.json" \
  --output-markdown "$project_root/ELF_SEMANTIC_SCALING_LADDER_2026-09-02.md"
