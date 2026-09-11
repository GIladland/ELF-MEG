#!/usr/bin/env bash
#SBATCH --job-name=fmri_package
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=submit-jobs/package_fmri_t5_single_checkpoint-%j.out
#SBATCH --error=submit-jobs/package_fmri_t5_single_checkpoint-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=${PYTHON_BIN:-$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python}
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${METRICS_JSON:?METRICS_JSON is required}"
: "${OUTPUT_CHECKPOINT:?OUTPUT_CHECKPOINT is required}"
: "${SYSTEM_NAME:?SYSTEM_NAME is required}"

cd "$project_root"
"$python_bin" scripts/package_fmri_t5_single_checkpoint.py \
  --checkpoint "$CHECKPOINT" \
  --metrics-json "$METRICS_JSON" \
  --output "$OUTPUT_CHECKPOINT" \
  --name "$SYSTEM_NAME"
