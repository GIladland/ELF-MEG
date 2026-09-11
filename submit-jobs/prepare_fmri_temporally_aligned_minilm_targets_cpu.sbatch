#!/usr/bin/env bash
#SBATCH --job-name=fmri_time_cpu
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=submit-jobs/prepare_fmri_temporally_aligned_minilm_targets_cpu-%j.out
#SBATCH --error=submit-jobs/prepare_fmri_temporally_aligned_minilm_targets_cpu-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export TARGET_DEVICE=cpu
exec bash "$project_root/submit-jobs/prepare_fmri_temporally_aligned_minilm_targets.sbatch"
