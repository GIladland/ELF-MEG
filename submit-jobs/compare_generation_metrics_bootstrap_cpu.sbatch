#!/usr/bin/env bash
#SBATCH --job-name=fmri_bootstrap
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=submit-jobs/compare_generation_metrics_bootstrap_cpu-%j.out
#SBATCH --error=submit-jobs/compare_generation_metrics_bootstrap_cpu-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
PYTHON_BIN=${PYTHON_BIN:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118/bin/python}
: "${BASELINE_JSON:?BASELINE_JSON is required}"
: "${CANDIDATE_JSON:?CANDIDATE_JSON is required}"
: "${OUTPUT_JSON:?OUTPUT_JSON is required}"

args=(
  --baseline "$BASELINE_JSON"
  --candidate "$CANDIDATE_JSON"
  --samples "${BOOTSTRAP_SAMPLES:-50000}"
  --seed "${BOOTSTRAP_SEED:-49}"
  --output "$OUTPUT_JSON"
)
if [[ -n ${BASELINE_BERTSCORE_JSON:-} || -n ${CANDIDATE_BERTSCORE_JSON:-} ]]; then
  : "${BASELINE_BERTSCORE_JSON:?both BERTScore JSON paths are required}"
  : "${CANDIDATE_BERTSCORE_JSON:?both BERTScore JSON paths are required}"
  args+=(
    --baseline-bertscore "$BASELINE_BERTSCORE_JSON"
    --candidate-bertscore "$CANDIDATE_BERTSCORE_JSON"
  )
fi

cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/compare_generation_metrics_bootstrap.py "${args[@]}"
