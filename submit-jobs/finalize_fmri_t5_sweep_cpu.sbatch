#!/usr/bin/env bash
#SBATCH --job-name=fmri_sweep_report
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=submit-jobs/finalize_fmri_t5_sweep_cpu-%j.out
#SBATCH --error=submit-jobs/finalize_fmri_t5_sweep_cpu-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=${PYTHON_BIN:-$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python}
: "${SWEEP_JSON:?SWEEP_JSON is required}"
: "${OUTPUT_PREFIX:?OUTPUT_PREFIX is required}"
text_npz=${TEXT_NPZ:-$data_root/minilm/knowntext_unseenbrain_v1/oracle_exact_minilm384_train_val.npz}

cd "$project_root"
export PYTHONPATH="$project_root:$project_root/src:$project_root/scripts:${PYTHONPATH:-}"

"$python_bin" scripts/extract_fmri_t5_sweep_record.py \
  --sweep-json "$SWEEP_JSON" \
  --record-index 0 \
  --output "${OUTPUT_PREFIX}_record0_metrics.json"

"$python_bin" scripts/rank_fmri_t5_sweep_informative.py \
  --sweep-json "$SWEEP_JSON" \
  --text-npz "$text_npz" \
  --output "${OUTPUT_PREFIX}_informative.json"

echo "Record and informative reports written for $SWEEP_JSON"
