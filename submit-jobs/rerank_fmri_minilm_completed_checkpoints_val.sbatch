#!/usr/bin/env bash
#SBATCH --job-name=fmri_done_ensemble
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=submit-jobs/rerank_fmri_minilm_completed_checkpoints_val-%j.out
#SBATCH --error=submit-jobs/rerank_fmri_minilm_completed_checkpoints_val-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
RUNS_ROOT=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
ENV_ROOT=${ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
PYTHON_BIN=${PYTHON_BIN:-$ENV_ROOT/bin/python}
BERTSCORE_PACKAGE_ROOT=${BERTSCORE_PACKAGE_ROOT:-$DATA_ROOT/python-packages/bert-score-0.3.13}
OUTPUT_DIR=$RUNS_ROOT/fmri_minilm_stage2_completed_checkpoint_ensemble_semantic_rerank_val266_20260830

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

source_npz=$RUNS_ROOT/fmri_minilm_stage2_sampler_seed7_cfg0p5_steps32_val266_20260830/eval_semantic_predictions.npz
if [[ ! -f "$source_npz" ]]; then
  echo "Missing model-predicted semantic archive: $source_npz" >&2
  exit 1
fi

baseline_run=$RUNS_ROOT/fmri_minilm_rawbrain_e2e_projector_long100_resdelaytok_h1024_l2_h8_storybal_b32_seed49_20260830
residual_run=$RUNS_ROOT/fmri_minilm_rawbrain_e2e_projector_long100_resmlp_h512_storybal_b32_seed49_20260830
candidate_files=()
while IFS= read -r path; do
  candidate_files+=("$path")
done < <(
  find \
    "$baseline_run" \
    "$residual_run" \
    "$RUNS_ROOT"/fmri_minilm_long100_*_20260830 \
    -maxdepth 1 -type f -name 'eval_step_*.json' -print | sort
)
if (( ${#candidate_files[@]} < 50 )); then
  echo "Unexpectedly small completed-checkpoint pool: ${#candidate_files[@]}" >&2
  exit 1
fi

metrics_args=()
for metrics in "${candidate_files[@]}"; do
  metrics_args+=(--metrics "$metrics")
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

printf '%s\n' "${candidate_files[@]}" > "$OUTPUT_DIR/candidate_files.txt"
echo "Completed-checkpoint semantic reranking written to $OUTPUT_DIR"
