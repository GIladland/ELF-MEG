#!/usr/bin/env bash
#SBATCH --job-name=fmri_long_report
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/report_fmri_long_program-%j.out
#SBATCH --error=submit-jobs/report_fmri_long_program-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
RUNS_ROOT=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
REPORT_DIR=${REPORT_DIR:-$RUNS_ROOT/fmri_long_program_report_20260830}
ENV_ROOT=${ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
PYTHON_BIN=${PYTHON_BIN:-$ENV_ROOT/bin/python}
BERTSCORE_PACKAGE_ROOT=${BERTSCORE_PACKAGE_ROOT:-$DATA_ROOT/python-packages/bert-score-0.3.13}
INCLUDE_ORACLE=${INCLUDE_ORACLE:-1}

export HF_HOME=${HF_HOME:-$DATA_ROOT/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export PYTHONNOUSERSITE=1
export PYTHONPATH=$BERTSCORE_PACKAGE_ROOT${PYTHONPATH:+:$PYTHONPATH}

mkdir -p "$REPORT_DIR" "$BERTSCORE_PACKAGE_ROOT" "$DATA_ROOT/pip-cache"
if [[ ! -f "$BERTSCORE_PACKAGE_ROOT/bert_score/__init__.py" ]]; then
  PIP_CACHE_DIR=$DATA_ROOT/pip-cache "$PYTHON_BIN" -m pip install \
    --no-user --no-deps --target "$BERTSCORE_PACKAGE_ROOT" bert-score==0.3.13
fi

cd "$PROJECT_ROOT"

shopt -s nullglob
run_dirs=(
  "$RUNS_ROOT"/fmri_minilm_long100_*
  "$RUNS_ROOT"/fmri_minilm_rawbrain_e2e_projector_long100_*
  "$RUNS_ROOT"/fmri_minilm_stage2_*_from_delaytok_best_*
  "$RUNS_ROOT"/fmri_minilm_stage2_sampler_*_val266_20260830
  "$RUNS_ROOT"/fmri_rawbrain_e2e_*_20260830*
)
if [[ "$INCLUDE_ORACLE" == "1" ]]; then
  run_dirs+=("$RUNS_ROOT"/fmri_knowntext_*_20260830)
fi

for run_dir in "${run_dirs[@]}"; do
  metrics=$run_dir/best_metrics.json
  bert=$run_dir/best_metrics.bertscore.json
  if [[ -f "$metrics" ]]; then
    lock=$run_dir/.best_metrics.bertscore.lock
    (
      flock 9
      # best_metrics.json can be replaced whenever a later checkpoint wins.
      # Recompute rather than silently pairing a new generation with stale BERTScore.
      if [[ ! -f "$bert" || "$metrics" -nt "$bert" ]]; then
        bert_tmp=$bert.tmp.${SLURM_JOB_ID:-manual}
        "$PYTHON_BIN" scripts/add_bertscore_to_metrics.py \
          --metrics-json "$metrics" \
          --output "$bert_tmp" \
          --batch-size 16 \
          --device cuda \
          --top-k 10
        mv "$bert_tmp" "$bert"
      fi
    ) 9>"$lock"
  fi
done

collector_args=()
if [[ "$INCLUDE_ORACLE" != "1" ]]; then
  collector_args+=(--exclude-known-text-oracle)
fi

"$PYTHON_BIN" scripts/collect_fmri_long_program_metrics.py \
  --runs-root "$RUNS_ROOT" \
  --output-json "$REPORT_DIR/metrics.json" \
  --output-csv "$REPORT_DIR/metrics.csv" \
  --output-markdown "$REPORT_DIR/REPORT.md" \
  "${collector_args[@]}"

echo "Report written to $REPORT_DIR"
