#!/usr/bin/env bash
#SBATCH --job-name=fmri_knntext
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=submit-jobs/analyze_fmri_oof_knn_text-%j.out
#SBATCH --error=submit-jobs/analyze_fmri_oof_knn_text-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
archive=${ARCHIVE:-$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz}
output=${OUTPUT:-/data/engs-pnpl/glandau/elf-runs/fmri_minilm_t5_prefix_20260831/oof_knn_text_audit.json}

cd "$project_root"
export PYTHONPATH="$project_root:$project_root/src:$project_root/scripts:${PYTHONPATH:-}"
"$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python" \
  scripts/analyze_fmri_oof_knn_text.py \
  --archive "$archive" \
  --output "$output" \
  --seed 49
