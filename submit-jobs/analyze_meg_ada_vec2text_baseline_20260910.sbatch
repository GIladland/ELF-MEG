#!/usr/bin/env bash
#SBATCH --job-name=meg_v2t_perm
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-3%4
#SBATCH --output=submit-jobs/analyze_meg_ada_vec2text_baseline_20260910-%A_%a.out
#SBATCH --error=submit-jobs/analyze_meg_ada_vec2text_baseline_20260910-%A_%a.err

set -euo pipefail

project_root=/data/engs-pnpl/glandau/BrainDiffusion/ELF
data_root=/data/engs-pnpl/glandau/elf-cache
run_root=/data/engs-pnpl/glandau/elf-runs/meg_ada_vec2text_baseline_20260910
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
bert_package=$data_root/python-packages/bert-score-0.3.13
roberta=/home/glandau/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) label=predicted_qc4wyals_wordcap10; source_label=predicted_qc4wyals; suffix= ;;
  1) label=predicted_qc4wyals_native; source_label=predicted_qc4wyals; suffix=.native ;;
  2) label=exact_ada_ceiling_wordcap10; source_label=exact_ada_ceiling; suffix= ;;
  3) label=exact_ada_ceiling_native; source_label=exact_ada_ceiling; suffix=.native ;;
  *) exit 2 ;;
esac

metrics=$run_root/$source_label/vec2text_steps20_beam4_wordcap10${suffix}.t5retrieval.json
for required in "$metrics" "$python_bin" "$roberta/config.json"; do
  [[ -s "$required" ]] || { echo "Missing $required" >&2; exit 2; }
done

export PYTHONPATH=$bert_package${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=/home/glandau/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "$project_root"

"$python_bin" scripts/analyze_generation_permutation.py \
  --input-metrics-json "$metrics" \
  --run-tag "meg_ada_vec2text_${label}_steps20_beam4_20260910" \
  --permutations 100000 \
  --seed 20260910 \
  --bertscore \
  --bertscore-model-path "$roberta" \
  --bertscore-layers 17 \
  --bertscore-device cuda \
  --bertscore-batch-size 64 \
  --output-json "$run_root/${label}.permutation.json" \
  --output-csv "$run_root/${label}.ranked_generations.csv"
