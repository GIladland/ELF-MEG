#!/usr/bin/env bash
#SBATCH --job-name=fmri_bert107
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=submit-jobs/eval_fmri_best_test107_bertscore-%j.out
#SBATCH --error=submit-jobs/eval_fmri_best_test107_bertscore-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=${PYTHON_BIN:-$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python}
package_root=${BERTSCORE_PACKAGE_ROOT:-$data_root/python-packages/bert-score-0.3.13}
run_tag=${RUN_TAG:-eval_fmri_minilm_knowntext_oofmix1536_adapteronly_ep10_b32_seed49_20260829_unseenbrain_direct1536_adapteronly_test107_20260829}
run_dir=/data/engs-pnpl/glandau/elf-runs/$run_tag

export HF_HOME=${HF_HOME:-$data_root/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export PYTHONNOUSERSITE=1
export PYTHONPATH=$package_root${PYTHONPATH:+:$PYTHONPATH}

mkdir -p "$package_root" "$data_root/pip-cache"
if [[ ! -f "$package_root/bert_score/__init__.py" ]]; then
  PIP_CACHE_DIR=$data_root/pip-cache "$python_bin" -m pip install \
    --no-user --no-deps --target "$package_root" bert-score==0.3.13
fi

cd "$project_root"
"$python_bin" scripts/add_bertscore_to_metrics.py \
  --metrics-json "$run_dir/best_metrics.json" \
  --output "$run_dir/bertscore_roberta_large_rescaled.json" \
  --lang en \
  --batch-size 16 \
  --device cuda \
  --top-k 10
