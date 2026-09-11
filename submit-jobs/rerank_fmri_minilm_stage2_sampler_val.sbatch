#!/usr/bin/env bash
#SBATCH --job-name=fmri_s2_rerank
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/rerank_fmri_minilm_stage2_sampler_val-%j.out
#SBATCH --error=submit-jobs/rerank_fmri_minilm_stage2_sampler_val-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
RUNS_ROOT=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
ENV_ROOT=${ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
PYTHON_BIN=${PYTHON_BIN:-$ENV_ROOT/bin/python}
BERTSCORE_PACKAGE_ROOT=${BERTSCORE_PACKAGE_ROOT:-$DATA_ROOT/python-packages/bert-score-0.3.13}
OUTPUT_DIR=$RUNS_ROOT/fmri_minilm_stage2_sampler_semantic_rerank12_val266_20260830

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

source_run=$RUNS_ROOT/fmri_minilm_stage2_sampler_seed7_cfg0p5_steps32_val266_20260830
source_npz=$source_run/eval_semantic_predictions.npz
if [[ ! -f "$source_npz" ]]; then
  echo "Missing model-predicted semantic archive: $source_npz" >&2
  exit 1
fi

metrics_args=()
for cfg in 0p5 1p0 1p5; do
  for seed in 7 23 49 101; do
    metrics=$RUNS_ROOT/fmri_minilm_stage2_sampler_seed${seed}_cfg${cfg}_steps32_val266_20260830/best_metrics.json
    if [[ ! -f "$metrics" ]]; then
      echo "Missing sampler metrics: $metrics" >&2
      exit 1
    fi
    metrics_args+=(--metrics "$metrics")
  done
done

cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/rerank_generated_candidates_by_semantic.py \
  --source-npz "$source_npz" \
  "${metrics_args[@]}" \
  --output "$OUTPUT_DIR/best_metrics.json" \
  --device cuda \
  --local-files-only

"$PYTHON_BIN" scripts/add_bertscore_to_metrics.py \
  --metrics-json "$OUTPUT_DIR/best_metrics.json" \
  --output "$OUTPUT_DIR/best_metrics.bertscore.json" \
  --batch-size 16 \
  --device cuda \
  --top-k 10

echo "Leakage-safe semantic reranking written to $OUTPUT_DIR"
