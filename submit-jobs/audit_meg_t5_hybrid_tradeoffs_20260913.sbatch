#!/usr/bin/env bash
#SBATCH --job-name=meg_t5traudit
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-1%2
#SBATCH --output=submit-jobs/audit_meg_t5_hybrid_tradeoffs_20260913-%A_%a.out
#SBATCH --error=submit-jobs/audit_meg_t5_hybrid_tradeoffs_20260913-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
runs_root=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
bert_package=$data_root/python-packages/bert-score-0.3.13
roberta=/home/glandau/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59
checkpoint=$runs_root/qc4wyals_elfb_dascoli_lora8_mid_ep40_seed49_20260911/best.pt

inputs=(
  "$runs_root/qc4wyals_elfb_t5confcopy_sem0.25_threshold0.2_flowend0.25_seed49_20260912_lexcapfix/best_metrics.json"
  "$runs_root/qc4wyals_elfb_t5confcopy_sem1.0_threshold0.2_flowend0.25_seed49_20260912_lexcapfix/best_metrics.json"
)
run_tags=(
  meg_t5_elfb_threshold0p2_sem0p25_flowend0p25
  meg_t5_elfb_threshold0p2_sem1p0_flowend0p25
)
output_names=(
  meg_t5_elfb_tradeoff_sem0p25_audit_20260913
  meg_t5_elfb_tradeoff_sem1p0_audit_20260913
)

task_id=${SLURM_ARRAY_TASK_ID:-0}
input=${inputs[$task_id]}
run_tag=${run_tags[$task_id]}
output_dir=$runs_root/${output_names[$task_id]}

for required in "$python_bin" "$input" "$checkpoint" "$roberta/config.json"; do
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
  --run-tag "$run_tag" \
  --checkpoint "$checkpoint" \
  --permutations 100000 \
  --seed 20260913 \
  --bertscore \
  --bertscore-model-path "$roberta" \
  --bertscore-layers 17 \
  --bertscore-device cuda \
  --bertscore-batch-size 64 \
  --output-json "$output_dir/permutation.json" \
  --output-csv "$output_dir/ranked_generations.csv"
