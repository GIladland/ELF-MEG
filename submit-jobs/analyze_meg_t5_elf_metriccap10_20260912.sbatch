#!/usr/bin/env bash
#SBATCH --job-name=meg_cap10audit
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-3%4
#SBATCH --output=submit-jobs/analyze_meg_t5_elf_metriccap10_20260912-%A_%a.out
#SBATCH --error=submit-jobs/analyze_meg_t5_elf_metriccap10_20260912-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
runs_root=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
bert_package=$data_root/python-packages/bert-score-0.3.13
roberta=/home/glandau/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59
input_dir=$project_root/meg_t5_elf_gap_closure_20260912
output_dir=$runs_root/meg_t5_elf_metriccap10_audit_20260912

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) label=t5_direct_projector ;;
  1) label=lexical_composite ;;
  2) label=content_specialist ;;
  3) label=retrieval_specialist ;;
  *) exit 2 ;;
esac

metrics=$input_dir/${label}_metriccap10.json
for required in "$python_bin" "$metrics" "$roberta/config.json"; do
  [[ -f "$required" ]] || { echo "Missing $required" >&2; exit 2; }
done
mkdir -p "$output_dir"
export PYTHONPATH=$bert_package${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=/home/glandau/.cache/huggingface
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

cd "$project_root"
exec "$python_bin" scripts/analyze_generation_permutation.py \
  --input-metrics-json "$metrics" \
  --run-tag "${label}_metriccap10" \
  --permutations 100000 \
  --seed 20260912 \
  --bertscore \
  --bertscore-model-path "$roberta" \
  --bertscore-layers 17 \
  --bertscore-device cuda \
  --bertscore-batch-size 64 \
  --output-json "$output_dir/${label}.json" \
  --output-csv "$output_dir/${label}_ranked_generations.csv"
