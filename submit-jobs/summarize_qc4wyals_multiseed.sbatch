#!/usr/bin/env bash
#SBATCH --job-name=qc4w_sum5
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=submit-jobs/summarize_qc4wyals_multiseed-%j.out
#SBATCH --error=submit-jobs/summarize_qc4wyals_multiseed-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=${PYTHON_BIN:-$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python}
: "${SELECTED_TAG:?Set SELECTED_TAG to the completed multiseed evaluation tag}"

eval_split=${EVAL_SPLIT:-val}
case "$eval_split" in
  val) eval_rows=110 ;;
  test) eval_rows=26 ;;
  *) echo "EVAL_SPLIT must be val or test, got: $eval_split" >&2; exit 2 ;;
esac

run_glob=/data/engs-pnpl/glandau/elf-runs/eval_qc4wyals_${eval_split}${eval_rows}_${SELECTED_TAG}_seed*_20260901
output_json=/data/engs-pnpl/glandau/elf-runs/qc4wyals_${eval_split}${eval_rows}_${SELECTED_TAG}_multiseed_summary_20260901.json

cd "$project_root"
exec "$python_bin" scripts/summarize_multiseed_generation.py \
  "$run_glob" \
  --output-json "$output_json"
