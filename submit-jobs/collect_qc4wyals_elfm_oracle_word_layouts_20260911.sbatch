#!/usr/bin/env bash
#SBATCH --job-name=elfm_oracollect
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=submit-jobs/collect_qc4wyals_elfm_oracle_word_layouts_20260911-%j.out
#SBATCH --error=submit-jobs/collect_qc4wyals_elfm_oracle_word_layouts_20260911-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
runs_root=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
python_bin=${PYTHON_BIN:-/data/engs-pnpl/glandau/elf-cache/conda-envs/braindiffusion-elf-torch271cu118/bin/python}
output_dir=$runs_root/meg_elfm_lexwinner_oracleword_layout_summary_20260911

cd "$project_root"
exec "$python_bin" scripts/collect_meg_oracle_word_ceiling.py \
  --runs-root "$runs_root" --output-dir "$output_dir"
