#!/usr/bin/env bash
#SBATCH --job-name=meg_t5_eval
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --array=0-9%3
#SBATCH --output=submit-jobs/eval_meg_ada_t5_prefix_ladder_20260907-%A_%a.out
#SBATCH --error=submit-jobs/eval_meg_ada_t5_prefix_ladder_20260907-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
env_root=$data_root/conda-envs/braindiffusion-elf-torch271cu118
python_bin=$env_root/bin/python
root=${ROOT_OVERRIDE:-/data/engs-pnpl/glandau/elf-runs/meg_ada_t5_prefix_20260907}

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) metrics=$root/selected_oracle/oracle_best_metrics.json ;;
  1) metrics=$root/raw_frozen_oracle/best_metrics.json ;;
  2) metrics=$root/adapter_h1024/best_metrics.json ;;
  3) metrics=$root/adapter_h2048_content_pair/best_metrics.json ;;
  4) metrics=$root/adapter_h1024_lora4/best_metrics.json ;;
  5) metrics=$root/adapter_h1024_lora8/best_metrics.json ;;
  6) metrics=$root/joint_projector_adapter/best_metrics.json ;;
  7) metrics=$root/joint_projector_adapter_lora4_content_pair/best_metrics.json ;;
  8) metrics=$root/direct_projector/best_metrics.json ;;
  9) metrics=$root/lora4_only/best_metrics.json ;;
  *) exit 2 ;;
esac

if [[ ! -s "$metrics" ]]; then
  echo "No completed metrics for task ${SLURM_ARRAY_TASK_ID}: $metrics" >&2
  exit 0
fi

export HF_HOME=${HF_HOME:-$data_root/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export PYTHONNOUSERSITE=1
export PYTHONPATH=$data_root/python-packages/bert-score-0.3.13${PYTHONPATH:+:$PYTHONPATH}
cd "$project_root"

"$python_bin" scripts/add_t5_retrieval_to_metrics.py \
  --metrics-json "$metrics" \
  --output "${metrics%.json}.t5retrieval.json" \
  --device cuda \
  --local-files-only

"$python_bin" scripts/add_bertscore_to_metrics.py \
  --metrics-json "$metrics" \
  --output "${metrics%.json}.bertscore_rescaled.json" \
  --batch-size 32 \
  --device cuda \
  --top-k 10

"$python_bin" scripts/add_bertscore_to_metrics.py \
  --metrics-json "$metrics" \
  --output "${metrics%.json}.bertscore_raw.json" \
  --batch-size 32 \
  --device cuda \
  --top-k 10 \
  --no-rescale-with-baseline
