#!/usr/bin/env bash
#SBATCH --job-name=meg_t5elfsel
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=submit-jobs/rerank_meg_t5_elf_roberta_oof_mapper_20260912-%j.out
#SBATCH --error=submit-jobs/rerank_meg_t5_elf_roberta_oof_mapper_20260912-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
runs_root=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
interface=$data_root/tang_themoth_qc4wyals_nested_oof_20260908/interface
oof_input=$interface/qc4wyals_oof_predicted_train_valtail_ada002.npz
exact_alignment=$interface/qc4wyals_oof_exact_alignment_train_valtail_ada002.npz
roberta=/home/glandau/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59
candidate_dir=$project_root/meg_t5_elf_gap_closure_20260912
output_dir=$runs_root/meg_t5_elf_roberta_oof_mapper_rerank_val110_20260912

for required in "$python_bin" "$oof_input" "$exact_alignment" "$roberta/config.json" \
  "$candidate_dir/t5_direct_projector_metriccap10.json" \
  "$candidate_dir/lexical_composite_metriccap10.json" \
  "$candidate_dir/content_specialist_metriccap10.json" \
  "$candidate_dir/retrieval_specialist_metriccap10.json"; do
  [[ -f "$required" ]] || { echo "Missing $required" >&2; exit 2; }
done
mkdir -p "$output_dir"
export HF_HOME=/home/glandau/.cache/huggingface
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd "$project_root"
exec "$python_bin" scripts/rerank_candidates_by_roberta_mapper.py \
  --train-npz "$exact_alignment" \
  --train-input-npz "$oof_input" \
  --train-input-start 0 \
  --train-rows 2652 \
  --val-semantic-npz "$oof_input" \
  --val-input-start 2652 \
  --val-rows 110 \
  --input-delay-blocks 1 \
  --metrics "$candidate_dir/t5_direct_projector_metriccap10.json" \
  --metrics "$candidate_dir/lexical_composite_metriccap10.json" \
  --metrics "$candidate_dir/content_specialist_metriccap10.json" \
  --metrics "$candidate_dir/retrieval_specialist_metriccap10.json" \
  --model-name "$roberta" \
  --layer 17 \
  --batch-size 48 \
  --max-length 64 \
  --device cuda \
  --local-files-only \
  --cache "$output_dir/train2652_roberta_large_layer17_mean_fp32.npy" \
  --output "$output_dir/best_metrics.json" \
  --length-penalty 0.0001 \
  --length-penalty 0.0002 \
  --length-penalty 0.0005 \
  --length-penalty 0.001 \
  --length-penalty 0.002 \
  --length-penalty 0.005 \
  --seed 49
