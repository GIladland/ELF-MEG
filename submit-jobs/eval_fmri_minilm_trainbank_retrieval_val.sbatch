#!/usr/bin/env bash
#SBATCH --job-name=fmri_trainbank
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=submit-jobs/eval_fmri_minilm_trainbank_retrieval_val-%j.out
#SBATCH --error=submit-jobs/eval_fmri_minilm_trainbank_retrieval_val-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
RUNS_ROOT=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
ENV_ROOT=${ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
PYTHON_BIN=${PYTHON_BIN:-$ENV_ROOT/bin/python}
BERTSCORE_PACKAGE_ROOT=${BERTSCORE_PACKAGE_ROOT:-$DATA_ROOT/python-packages/bert-score-0.3.13}
OUTPUT_DIR=$RUNS_ROOT/fmri_minilm_stage2_trainbank11725_retrieval_val266_20260830

export HF_HOME=${HF_HOME:-$DATA_ROOT/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export PYTHONNOUSERSITE=1
export PYTHONPATH=$BERTSCORE_PACKAGE_ROOT${PYTHONPATH:+:$PYTHONPATH}

mkdir -p "$OUTPUT_DIR" "$BERTSCORE_PACKAGE_ROOT" "$DATA_ROOT/pip-cache"
if [[ ! -f "$BERTSCORE_PACKAGE_ROOT/bert_score/__init__.py" ]]; then
  PIP_CACHE_DIR=$DATA_ROOT/pip-cache "$PYTHON_BIN" -m pip install \
    --no-user --no-deps --target "$BERTSCORE_PACKAGE_ROOT" bert-score==0.3.13
fi

cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/evaluate_trainbank_semantic_retrieval.py \
  --query-npz "$RUNS_ROOT/fmri_minilm_stage2_sampler_seed7_cfg0p5_steps32_val266_20260830/eval_semantic_predictions.npz" \
  --bank-npz "$DATA_ROOT/minilm/knowntext_unseenbrain_v1/oracle_exact_minilm384_train_val.npz" \
  --bank-split train \
  --top-k 5 \
  --output "$OUTPUT_DIR/best_metrics.json"

"$PYTHON_BIN" scripts/add_bertscore_to_metrics.py \
  --metrics-json "$OUTPUT_DIR/best_metrics.json" \
  --output "$OUTPUT_DIR/best_metrics.bertscore.json" \
  --batch-size 16 \
  --device cuda \
  --top-k 10

echo "Strict train-bank retrieval report written to $OUTPUT_DIR"
