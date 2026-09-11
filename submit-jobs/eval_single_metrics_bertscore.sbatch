#!/usr/bin/env bash
#SBATCH --job-name=fmri_bert_one
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/eval_single_metrics_bertscore-%j.out
#SBATCH --error=submit-jobs/eval_single_metrics_bertscore-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
ENV_ROOT=${ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
PYTHON_BIN=${PYTHON_BIN:-$ENV_ROOT/bin/python}
BERTSCORE_PACKAGE_ROOT=${BERTSCORE_PACKAGE_ROOT:-$DATA_ROOT/python-packages/bert-score-0.3.13}
BERTSCORE_DEVICE=${BERTSCORE_DEVICE:-cuda}
: "${METRICS_JSON:?METRICS_JSON is required}"
OUTPUT_JSON=${OUTPUT_JSON:-${METRICS_JSON%.json}.bertscore.json}

export HF_HOME=${HF_HOME:-$DATA_ROOT/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export PYTHONNOUSERSITE=1
export PYTHONPATH=$BERTSCORE_PACKAGE_ROOT${PYTHONPATH:+:$PYTHONPATH}

cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/add_bertscore_to_metrics.py \
  --metrics-json "$METRICS_JSON" \
  --output "$OUTPUT_JSON" \
  --batch-size 16 \
  --device "$BERTSCORE_DEVICE" \
  --top-k 10

echo "BERTScore written to $OUTPUT_JSON"
