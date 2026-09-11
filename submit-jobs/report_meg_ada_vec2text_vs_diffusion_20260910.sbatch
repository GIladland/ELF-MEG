#!/usr/bin/env bash
#SBATCH --job-name=meg_v2t_report
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=submit-jobs/report_meg_ada_vec2text_vs_diffusion_20260910-%j.out
#SBATCH --error=submit-jobs/report_meg_ada_vec2text_vs_diffusion_20260910-%j.err

set -euo pipefail

project_root=/data/engs-pnpl/glandau/BrainDiffusion/ELF
run_root=/data/engs-pnpl/glandau/elf-runs/meg_ada_vec2text_baseline_20260910
diffusion=/data/engs-pnpl/glandau/elf-runs/meg_three_candidate_permutation_20260910/lexical_composite.json
python_bin=/data/engs-pnpl/glandau/elf-cache/conda-envs/braindiffusion-elf-torch271cu118/bin/python
cd "$project_root"

"$python_bin" scripts/report_meg_vec2text_vs_diffusion.py \
  --diffusion-json "$diffusion" \
  --predicted-capped-json "$run_root/predicted_qc4wyals_wordcap10.permutation.json" \
  --predicted-native-json "$run_root/predicted_qc4wyals_native.permutation.json" \
  --exact-capped-json "$run_root/exact_ada_ceiling_wordcap10.permutation.json" \
  --exact-native-json "$run_root/exact_ada_ceiling_native.permutation.json" \
  --output-markdown "$run_root/MEG_ADA_VEC2TEXT_VS_ELF_DIFFUSION_2026-09-10.md" \
  --output-csv "$run_root/actual_vs_permutation_summary.csv"
