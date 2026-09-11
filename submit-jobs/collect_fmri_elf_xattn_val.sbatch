#!/usr/bin/env bash
#SBATCH --job-name=xattn_collect
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=submit-jobs/collect_fmri_elf_xattn_val-%j.out
#SBATCH --error=submit-jobs/collect_fmri_elf_xattn_val-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
run_root=${RUN_ROOT:-/data/engs-pnpl/glandau/elf-runs}
output_json=${OUTPUT_JSON:-$run_root/fmri_elf_xattn_val_leaderboard_20260831.json}

exec python "$project_root/scripts/collect_fmri_xattn_results.py" \
  --run-glob "$run_root/fmri_minilm_elf_xattn_*_from_recover2_step1000_20260831*" \
  --output-json "$output_json"
