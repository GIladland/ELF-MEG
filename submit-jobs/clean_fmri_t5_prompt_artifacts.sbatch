#!/usr/bin/env bash
#SBATCH --job-name=fmri_cleanup
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=submit-jobs/clean_fmri_t5_prompt_artifacts-%j.out
#SBATCH --error=submit-jobs/clean_fmri_t5_prompt_artifacts-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
: "${METRICS_JSON:?METRICS_JSON is required}"
: "${OUTPUT_JSON:?OUTPUT_JSON is required}"
cd "$project_root"
export PYTHONPATH="$project_root:$project_root/src:$project_root/scripts:${PYTHONPATH:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$data_root/conda-envs/braindiffusion-elf-torch271cu118"
python scripts/clean_fmri_t5_prompt_artifacts.py \
  --metrics-json "$METRICS_JSON" \
  --output "$OUTPUT_JSON"
