#!/usr/bin/env bash
#SBATCH --job-name=qc4w_bert5
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --array=0-4%5
#SBATCH --output=submit-jobs/eval_qc4wyals_multiseed_bertscore-%A_%a.out
#SBATCH --error=submit-jobs/eval_qc4wyals_multiseed_bertscore-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=${PYTHON_BIN:-$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python}
package_root=${BERTSCORE_PACKAGE_ROOT:-$data_root/python-packages/bert-score-0.3.13}
: "${SELECTED_TAG:?Set SELECTED_TAG to the multiseed evaluation tag}"
eval_split=${EVAL_SPLIT:-val}
case "$eval_split" in
  val) eval_rows=110 ;;
  test) eval_rows=26 ;;
  *) echo "EVAL_SPLIT must be val or test, got: $eval_split" >&2; exit 2 ;;
esac

seeds=(49 101 202 303 404)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
run_dir=/data/engs-pnpl/glandau/elf-runs/eval_qc4wyals_${eval_split}${eval_rows}_${SELECTED_TAG}_seed${seed}_20260901
metrics_json=$run_dir/best_metrics.json
output_json=$run_dir/best_metrics.bertscore.json
[[ -f "$metrics_json" ]] || { echo "Missing $metrics_json" >&2; exit 2; }

export HF_HOME=${HF_HOME:-$data_root/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export PYTHONNOUSERSITE=1
export PYTHONPATH=$package_root${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}

cd "$project_root"
exec "$python_bin" scripts/add_bertscore_to_metrics.py \
  --metrics-json "$metrics_json" \
  --output "$output_json" \
  --batch-size 8 \
  --device cpu \
  --top-k 10
