#!/usr/bin/env bash
#SBATCH --job-name=fmri_pres_report
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=submit-jobs/report_fmri_freshmem_semantic_preservation-%j.out
#SBATCH --error=submit-jobs/report_fmri_freshmem_semantic_preservation-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
snapshot_tag=${ORACLE_SNAPSHOT_TAG:-step024000}
run_root=/data/engs-pnpl/glandau/elf-runs
output_dir=$run_root/fmri_freshmem_semantic_preservation_${snapshot_tag}_selection_20260901/report

cd "$PROJECT_ROOT"
python scripts/report_fmri_semantic_preservation.py \
  --run-root "$run_root" \
  --snapshot-tag "$snapshot_tag" \
  --output-dir "$output_dir"
