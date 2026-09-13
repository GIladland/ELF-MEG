#!/usr/bin/env bash
#SBATCH --job-name=qc4_winroll
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --array=0-2%2
#SBATCH --output=submit-jobs/eval_qc4wyals_elfm_scattered_winner_rolls_20260912-%A_%a.out
#SBATCH --error=submit-jobs/eval_qc4wyals_elfm_scattered_winner_rolls_20260912-%A_%a.err

set -euo pipefail

rolls=(1 37 73)
task=${SLURM_ARRAY_TASK_ID:-0}
export CONTENT_BIAS_OVERRIDE=24.0
export CONTENT_BIAS_TAG_OVERRIDE=24p0
export ORDERED_BIAS_OVERRIDE=32.0
export ORDERED_BIAS_TAG_OVERRIDE=32p0
export POSITION_PRIOR_OVERRIDE=0.40
export POSITION_PRIOR_TAG_OVERRIDE=0p40
export GENERATION_SEED_OVERRIDE=49
export SEMANTIC_VAL_ROLL_OVERRIDE=${rolls[$task]}

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
exec bash "$project_root/submit-jobs/eval_qc4wyals_elfm_semantic_scattered_bias_20260912.sbatch"
