#!/usr/bin/env bash
#SBATCH --job-name=meg_t5b4audit
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=submit-jobs/audit_meg_t5_beam4_winner_20260912-%j.out
#SBATCH --error=submit-jobs/audit_meg_t5_beam4_winner_20260912-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
runs_root=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
bert_package=$data_root/python-packages/bert-score-0.3.13
roberta=/home/glandau/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59
input=$runs_root/meg_t5_source_confidence_val110_20260912/beam4_min10_max20_lp0p8.json
output_dir=$runs_root/meg_t5_beam4_winner_audit_20260912

for required in "$python_bin" "$input" "$roberta/config.json"; do
  [[ -f "$required" ]] || { echo "Missing $required" >&2; exit 2; }
done
mkdir -p "$output_dir"
export PYTHONPATH=$bert_package${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=/home/glandau/.cache/huggingface
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

cd "$project_root"
"$python_bin" scripts/add_t5_retrieval_to_metrics.py \
  --metrics-json "$input" \
  --output "$output_dir/best_metrics.retrieval.json" \
  --device cuda \
  --local-files-only

exec "$python_bin" scripts/analyze_generation_permutation.py \
  --input-metrics-json "$output_dir/best_metrics.retrieval.json" \
  --run-tag meg_t5_direct_beam4_min10_max20_lp0p8 \
  --checkpoint "$runs_root/meg_ada_t5_prefix_20260907/direct_projector/best.pt" \
  --permutations 100000 \
  --seed 20260912 \
  --bertscore \
  --bertscore-model-path "$roberta" \
  --bertscore-layers 17 \
  --bertscore-device cuda \
  --bertscore-batch-size 64 \
  --output-json "$output_dir/permutation.json" \
  --output-csv "$output_dir/ranked_generations.csv"
