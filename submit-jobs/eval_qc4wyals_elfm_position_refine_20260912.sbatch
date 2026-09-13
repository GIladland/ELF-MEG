#!/usr/bin/env bash
#SBATCH --job-name=qc4_posref
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --array=0-4%2
#SBATCH --output=submit-jobs/eval_qc4wyals_elfm_position_refine_20260912-%A_%a.out
#SBATCH --error=submit-jobs/eval_qc4wyals_elfm_position_refine_20260912-%A_%a.err

set -euo pipefail

values=(0.20 0.60 0.80 1.20 1.60)
tags=(0p20 0p60 0p80 1p20 1p60)
task=${SLURM_ARRAY_TASK_ID:-0}
export POSITION_PRIOR_OVERRIDE=${values[$task]}
export POSITION_PRIOR_TAG_OVERRIDE=${tags[$task]}
export ORDERED_BIAS_OVERRIDE=24.0
export ORDERED_BIAS_TAG_OVERRIDE=24p0

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
exec bash "$project_root/submit-jobs/eval_qc4wyals_elfm_semantic_combined_bias_20260912.sbatch"
