#!/usr/bin/env bash
#SBATCH --job-name=elfm_ctxpick
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=submit-jobs/select_qc4wyals_elfm_context_distill_20260906-%j.out
#SBATCH --error=submit-jobs/select_qc4wyals_elfm_context_distill_20260906-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
root=/data/engs-pnpl/glandau/elf-runs
output=$root/meg_elfm_ctxdistill_selected_20260906
python=${PYTHON_BIN:-/data/engs-pnpl/glandau/elf-cache/conda-envs/braindiffusion-elf-torch271cu118/bin/python}
mkdir -p "$output"
exec "$python" "$project_root/scripts/select_context_distillation_winner.py" \
  --run-dir "$root/meg_elfm_ctxdistill_exactteacher_output_only_ep100_seed49_20260906" \
  --run-dir "$root/meg_elfm_ctxdistill_exactteacher_projector_only_ep100_seed49_20260906" \
  --run-dir "$root/meg_elfm_ctxdistill_exactteacher_output_plus_projector_ep100_seed49_20260906" \
  --run-dir "$root/meg_elfm_ctxdistill_exactteacher_fullmeg_plus_projector_ep100_seed49_20260906" \
  --run-dir "$root/meg_elfm_ctxdistill_robustinit_exactteacher_outputprojector_ep100_seed49_20260906" \
  --output-checkpoint "$output/selected_context_adapter.pt" \
  --output-json "$output/selection.json"
