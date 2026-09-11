#!/usr/bin/env bash
#SBATCH --job-name=gen_perm
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=submit-jobs/analyze_single_generation_permutation_20260910-%j.out
#SBATCH --error=submit-jobs/analyze_single_generation_permutation_20260910-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
bert_package=$data_root/python-packages/bert-score-0.3.13
roberta=/home/glandau/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59

: "${INPUT_METRICS_JSON:?Set INPUT_METRICS_JSON}"
: "${OUTPUT_JSON:?Set OUTPUT_JSON}"
: "${OUTPUT_CSV:?Set OUTPUT_CSV}"

export PYTHONPATH=$bert_package${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=/home/glandau/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$project_root"
"$python_bin" scripts/analyze_generation_permutation.py \
  --input-metrics-json "$INPUT_METRICS_JSON" \
  --run-tag "${RUN_TAG:-$(basename "$INPUT_METRICS_JSON" .json)}" \
  --checkpoint "${CHECKPOINT:-}" \
  --permutations "${PERMUTATIONS:-100000}" \
  --seed "${SEED:-20260910}" \
  --bertscore \
  --bertscore-model-path "$roberta" \
  --bertscore-layers 17 \
  --bertscore-device cuda \
  --bertscore-batch-size 64 \
  --output-json "$OUTPUT_JSON" \
  --output-csv "$OUTPUT_CSV"
