#!/usr/bin/env bash
#SBATCH --job-name=mpnet_perm
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=submit-jobs/analyze_best_hybrid_mpnet_permutation_20260914-%j.out
#SBATCH --error=submit-jobs/analyze_best_hybrid_mpnet_permutation_20260914-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
audit_dir=$project_root/meg_t5_elf_gap_closure_20260912/arc_results/audit/meg_t5_elfb_confcopy_winner_audit_20260912
input_csv=$audit_dir/ranked_generations.csv
output_stem=$audit_dir/ranked_generations_mpnet_sentence_embedding_permutation

for required in "$python_bin" "$input_csv"; do
  [[ -f "$required" ]] || { echo "Missing required path: $required" >&2; exit 2; }
done

mkdir -p "$audit_dir" "$data_root/tmp"
export HF_HOME=$data_root/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export TMPDIR=$data_root/tmp

cd "$project_root"
exec "$python_bin" -u scripts/analyze_generation_permutation.py \
  --input-csv "$input_csv" \
  --run-tag meg_t5_elfb_confcopy_winner_mpnet \
  --permutations 10000 \
  --seed 20260914 \
  --sentence-embedding \
  --sentence-embedding-model sentence-transformers/all-mpnet-base-v2 \
  --sentence-embedding-device cuda \
  --sentence-embedding-batch-size 64 \
  --sentence-embedding-max-length 64 \
  --output-json "$output_stem.json" \
  --output-csv "$output_stem.csv"
