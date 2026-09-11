#!/usr/bin/env bash
#SBATCH --job-name=fmri_t5_retr
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=submit-jobs/add_t5_retrieval_to_metrics-%j.out
#SBATCH --error=submit-jobs/add_t5_retrieval_to_metrics-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
ENV_ROOT=${ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
: "${METRICS_JSON:?METRICS_JSON is required}"
OUTPUT_JSON=${OUTPUT_JSON:-${METRICS_JSON%.json}.t5retrieval.json}

export HF_HOME=${HF_HOME:-$DATA_ROOT/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export PYTHONNOUSERSITE=1

cd "$PROJECT_ROOT"
"$ENV_ROOT/bin/python" scripts/add_t5_retrieval_to_metrics.py \
  --metrics-json "$METRICS_JSON" \
  --output "$OUTPUT_JSON" \
  --device cuda \
  --local-files-only
