#!/usr/bin/env bash
#SBATCH --job-name=qc4_oofreport
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=submit-jobs/report_qc4wyals_nested_oof_elfb_gap_20260908-%j.out
#SBATCH --error=submit-jobs/report_qc4wyals_nested_oof_elfb_gap_20260908-%j.err

set -euo pipefail

brain_root=/data/engs-pnpl/glandau/BrainDiffusion/ELF
python_bin=/data/engs-pnpl/glandau/elf-cache/conda-envs/braindiffusion-elf-torch271cu118/bin/python
run_root=/data/engs-pnpl/glandau/elf-runs
oof_root=/data/engs-pnpl/glandau/elf-cache/tang_themoth_qc4wyals_nested_oof_20260908
report_root=$run_root/qc4wyals_nested_oof5_elfb_gap_report_20260908

mkdir -p "$report_root"
cd "$brain_root"
exec "$python_bin" scripts/report_meg_elfb_oof_gap.py \
  --oof-summary "$oof_root/interface/qc4wyals_nested_oof_summary.json" \
  --system "exact_ada=$run_root/eval_qc4wyals_val110_oofgap_exact_seed*_20260901" \
  --system "raw_meg=$run_root/eval_qc4wyals_val110_oofgap_raw_seed*_20260901" \
  --system "oof_flat=$run_root/eval_qc4wyals_val110_oofgap_flat_seed*_20260901" \
  --system "oof_residual=$run_root/eval_qc4wyals_val110_oofgap_residual_seed*_20260901" \
  --output-json "$report_root/gap_closure.json" \
  --output-md "$report_root/gap_closure.md"
