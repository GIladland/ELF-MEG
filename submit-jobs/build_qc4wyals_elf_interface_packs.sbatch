#!/usr/bin/env bash
#SBATCH --job-name=qc4w_packs
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=submit-jobs/build_qc4wyals_elf_interface_packs-%j.out
#SBATCH --error=submit-jobs/build_qc4wyals_elf_interface_packs-%j.err

set -euo pipefail

project_root=/data/engs-pnpl/glandau/BrainDiffusion/ELF
prediction_dir=/data/engs-pnpl/glandau/elf-cache/tang_themoth_qc4wyals_ada_v2
knowntext=/data/engs-pnpl/glandau/elf-cache/tang_themoth_apples_exact_ada002_textmid_seg10_v1/tang_apples_ada002_knowntext_all2788_testfirst.npz
output_dir=$prediction_dir/interface

cd "$project_root"
exec /data/engs-pnpl/glandau/conda/envs/meg2sem/bin/python \
  scripts/build_qc4wyals_elf_interface_packs.py \
  --prediction-dir "$prediction_dir" \
  --knowntext-npz "$knowntext" \
  --output-dir "$output_dir" \
  --residual-scales 0.5,1.0,1.5 \
  --seed 49
