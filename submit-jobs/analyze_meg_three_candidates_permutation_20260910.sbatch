#!/usr/bin/env bash
#SBATCH --job-name=meg_perm3
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-2%3
#SBATCH --output=submit-jobs/analyze_meg_three_candidates_permutation_20260910-%A_%a.out
#SBATCH --error=submit-jobs/analyze_meg_three_candidates_permutation_20260910-%A_%a.err

set -euo pipefail

project_root=/data/engs-pnpl/glandau/BrainDiffusion/ELF
data_root=/data/engs-pnpl/glandau/elf-cache
runs_root=/data/engs-pnpl/glandau/elf-runs
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
bert_package=$data_root/python-packages/bert-score-0.3.13
roberta=/home/glandau/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59
output_dir=$runs_root/meg_three_candidate_permutation_20260910

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    label=lexical_composite
    run=meg_elfm_ctxdistill_robustinit_exactteacher_outputprojector_ep100_seed49_20260906
    ;;
  1)
    label=content_specialist
    run=elf_archscale_meg_ada_x16_elfm_continue_const1e5_ep8_wcmean_seed7_20260906_resid_adapter_val110_wcmean_seed49_truee2e_output_projector_lora4last4_val110_wcmean_seed49
    ;;
  2)
    label=retrieval_specialist
    run=elf_archscale_meg_ada_x16_elfm_continue_const1e5_ep8_wcmean_seed7_20260906_resid_adapter_val110_wcmean_seed49_truee2e_allmeg_projector_lora4last4_val110_wcmean_seed49
    ;;
  *)
    exit 2
    ;;
esac

metrics=$runs_root/$run/best_metrics.json
checkpoint=$runs_root/$run/best_adapter.pt
for required in "$python_bin" "$metrics" "$checkpoint" "$roberta/config.json"; do
  [[ -f "$required" ]] || { echo "Missing $required" >&2; exit 2; }
done
mkdir -p "$output_dir"
export PYTHONPATH=$bert_package${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=/home/glandau/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$project_root"
"$python_bin" scripts/analyze_generation_permutation.py \
  --input-metrics-json "$metrics" \
  --run-tag "$run" \
  --checkpoint "$checkpoint" \
  --permutations 100000 \
  --seed 20260910 \
  --bertscore \
  --bertscore-model-path "$roberta" \
  --bertscore-layers 17 \
  --bertscore-device cuda \
  --bertscore-batch-size 64 \
  --output-json "$output_dir/${label}.json" \
  --output-csv "$output_dir/${label}_ranked_generations.csv"
