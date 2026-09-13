#!/usr/bin/env bash
#SBATCH --job-name=qc4_scatfine
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --array=0-6%2
#SBATCH --output=submit-jobs/eval_qc4wyals_elfm_scattered_boundary_20260912-%A_%a.out
#SBATCH --error=submit-jobs/eval_qc4wyals_elfm_scattered_boundary_20260912-%A_%a.err

set -euo pipefail

content_values=(24.0 24.0 24.0 24.0 25.0 26.0 28.0)
content_tags=(24p0 24p0 24p0 24p0 25p0 26p0 28p0)
ordered_values=(25.0 26.0 27.0 28.0 24.0 24.0 24.0)
ordered_tags=(25p0 26p0 27p0 28p0 24p0 24p0 24p0)
task=${SLURM_ARRAY_TASK_ID:-0}

export CONTENT_BIAS_OVERRIDE=${content_values[$task]}
export CONTENT_BIAS_TAG_OVERRIDE=${content_tags[$task]}
export ORDERED_BIAS_OVERRIDE=${ordered_values[$task]}
export ORDERED_BIAS_TAG_OVERRIDE=${ordered_tags[$task]}
export POSITION_PRIOR_OVERRIDE=0.40
export POSITION_PRIOR_TAG_OVERRIDE=0p40

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
exec bash "$project_root/submit-jobs/eval_qc4wyals_elfm_semantic_scattered_bias_20260912.sbatch"
