#!/usr/bin/env bash
#SBATCH --job-name=prep_meg_mini_ho
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=submit-jobs/prepare_meg_minilm_true_holdout-%j.out
#SBATCH --error=submit-jobs/prepare_meg_minilm_true_holdout-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
env_root=${TMP_ENV_ROOT:-$data_root/conda-envs/braindiffusion-elf-torch271cu118}
source_dir=$data_root/sherlock_sva/meg2sem

export HF_HOME=${HF_HOME:-$data_root/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export TOKENIZERS_PARALLELISM=false

source "$env_root/bin/activate"
cd "$project_root"

for split in val test; do
  if [[ "$split" == "val" ]]; then
    source_npz=$source_dir/sherlock1_ses11_run2_sva_no_coord_decl5to18_val_ada002_meg2sem_segment3000_fp16.npz
    output_npz=$source_dir/sherlock1_ses11_run2_sva_no_coord_decl5to18_val_minilm_meg2sem_segment3000_fp16.npz
  else
    source_npz=$source_dir/sherlock1_ses12_run2_sva_no_coord_decl5to18_test_ada002_meg2sem_segment3000_fp16.npz
    output_npz=$source_dir/sherlock1_ses12_run2_sva_no_coord_decl5to18_test_minilm_meg2sem_segment3000_fp16.npz
  fi

  python scripts/reembed_semantic_npz_sentences.py \
    --input-npz "$source_npz" \
    --output "$output_npz" \
    --input-key input_embeddings \
    --sentence-key sentence \
    --embedding-model-name sentence-transformers/all-MiniLM-L6-v2 \
    --pooling mean \
    --max-length 128 \
    --batch-size 128 \
    --device cuda \
    --local-files-only
done

