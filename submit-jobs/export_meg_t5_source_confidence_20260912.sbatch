#!/usr/bin/env bash
#SBATCH --job-name=meg_t5conf
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=submit-jobs/export_meg_t5_source_confidence_20260912-%j.out
#SBATCH --error=submit-jobs/export_meg_t5_source_confidence_20260912-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
runs_root=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
checkpoint=$runs_root/meg_ada_t5_prefix_20260907/direct_projector/best.pt
semantic_npz=$data_root/tang_themoth_qc4wyals_ada_v2/interface/qc4wyals_predicted_train_valtail_ada002.npz
output_dir=$runs_root/meg_t5_source_confidence_val110_20260912

for required in "$python_bin" "$checkpoint" "$semantic_npz"; do
  [[ -f "$required" ]] || { echo "Missing $required" >&2; exit 2; }
done
mkdir -p "$output_dir"
export HF_HOME=$data_root/huggingface TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

cd "$project_root"
exec "$python_bin" scripts/export_meg_t5_source_confidence.py \
  --checkpoint "$checkpoint" \
  --semantic-npz "$semantic_npz" \
  --output-json "$output_dir/beam4_min10_max20_lp0p8.json" \
  --val-start 2652 \
  --val-rows 110 \
  --num-beams 4 \
  --min-new-tokens 10 \
  --max-new-tokens 20 \
  --length-penalty 0.8 \
  --max-output-words 10 \
  --batch-size 55 \
  --device cuda \
  --seed 49
