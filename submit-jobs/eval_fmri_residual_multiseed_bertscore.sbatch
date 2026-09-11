#!/usr/bin/env bash
#SBATCH --job-name=fmri_bertval
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-5
#SBATCH --output=submit-jobs/eval_fmri_residual_multiseed_bertscore-%A_%a.out
#SBATCH --error=submit-jobs/eval_fmri_residual_multiseed_bertscore-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=${PYTHON_BIN:-$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python}
package_root=${BERTSCORE_PACKAGE_ROOT:-$data_root/python-packages/bert-score-0.3.13}
robustness_steps=${ROBUSTNESS_SAMPLING_STEPS:-16}

case "$SLURM_ARRAY_TASK_ID" in
  0) candidate=wer; seed=7 ;;
  1) candidate=wer; seed=23 ;;
  2) candidate=wer; seed=101 ;;
  3) candidate=content; seed=7 ;;
  4) candidate=content; seed=23 ;;
  5) candidate=content; seed=101 ;;
  *) exit 2 ;;
esac

if [[ "$candidate" == "wer" ]]; then
  source_run=${WER_SOURCE_RUN:-fmri_minilm_knowntext_oofmix1536_resmlp_h512_norm_lr5e4_inputonly_ep20_b32_seed49_20260829}
else
  source_run=${CONTENT_SOURCE_RUN:-fmri_minilm_knowntext_oofmix1536_resmlp_h512_lr1e4_inputonly_ep20_b32_seed49_20260829}
fi
run_tag=eval_${source_run}_predval266_steps${robustness_steps}_decodeseed${seed}_20260829
run_dir=/data/engs-pnpl/glandau/elf-runs/$run_tag

export HF_HOME=${HF_HOME:-$data_root/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export PYTHONNOUSERSITE=1
export PYTHONPATH=$package_root${PYTHONPATH:+:$PYTHONPATH}

if [[ ! -f "$run_dir/best_metrics.json" ]]; then
  echo "Missing completed generation metrics: $run_dir/best_metrics.json" >&2
  exit 1
fi

cd "$project_root"
"$python_bin" scripts/add_bertscore_to_metrics.py \
  --metrics-json "$run_dir/best_metrics.json" \
  --output "$run_dir/bertscore_roberta_large_rescaled.json" \
  --lang en \
  --batch-size 16 \
  --device cuda \
  --top-k 10
