#!/usr/bin/env bash
#SBATCH --job-name=fmri_mem_report
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=submit-jobs/report_fmri_knowntext_freshmem-%j.out
#SBATCH --error=submit-jobs/report_fmri_knowntext_freshmem-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
snapshot_tag=${ORACLE_SNAPSHOT_TAG:?Set ORACLE_SNAPSHOT_TAG}
oracle_train_tag=${ORACLE_TRAIN_TAG:-fmri_exact_minilm384_fresh_elfb_full_memorize_all12098_ep600_seed7_20260901}
baseline_eval_tag=${BASELINE_EVAL_TAG:-${oracle_train_tag}_interim024k}
output_dir=/data/engs-pnpl/glandau/elf-runs/fmri_knowntext_freshmem_${snapshot_tag}_selection_20260901/report

cd "$PROJECT_ROOT"
python scripts/report_fmri_knowntext_freshmem.py \
  --run-root /data/engs-pnpl/glandau/elf-runs \
  --snapshot-tag "$snapshot_tag" \
  --oracle-train-tag "$oracle_train_tag" \
  --baseline-eval-tag "$baseline_eval_tag" \
  --output-dir "$output_dir"
