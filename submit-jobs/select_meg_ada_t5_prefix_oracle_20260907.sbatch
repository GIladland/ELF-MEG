#!/usr/bin/env bash
#SBATCH --job-name=meg_t5_select
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=submit-jobs/select_meg_ada_t5_prefix_oracle_20260907-%j.out
#SBATCH --error=submit-jobs/select_meg_ada_t5_prefix_oracle_20260907-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
root=${ROOT_OVERRIDE:-/data/engs-pnpl/glandau/elf-runs/meg_ada_t5_prefix_20260907}
cd "$project_root"
exec python scripts/select_meg_t5_prefix_oracle.py \
  --root "$root" \
  --output-dir "$root/selected_oracle"
