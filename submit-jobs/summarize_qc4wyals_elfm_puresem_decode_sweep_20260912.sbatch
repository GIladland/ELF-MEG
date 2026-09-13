#!/usr/bin/env bash
#SBATCH --job-name=qc4_dec_sum
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=submit-jobs/summarize_qc4wyals_elfm_puresem_decode_sweep_20260912-%j.out
#SBATCH --error=submit-jobs/summarize_qc4wyals_elfm_puresem_decode_sweep_20260912-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
runs_root=/data/engs-pnpl/glandau/elf-runs
output_dir=$runs_root/meg_elfm_puresem_decode_sweep_20260912
mkdir -p "$output_dir"
cd "$project_root"
exec /usr/bin/python3 scripts/summarize_meg_puresem_decode_sweep.py \
  --runs-root "$runs_root" \
  --output-json "$output_dir/summary.json" \
  --output-csv "$output_dir/summary.csv"
