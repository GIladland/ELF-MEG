#!/usr/bin/env bash
#SBATCH --job-name=elfb_daseval
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --array=0-9%5
#SBATCH --output=submit-jobs/eval_qc4wyals_elfb_dascoli_source_matrix_20260911-%A_%a.out
#SBATCH --error=submit-jobs/eval_qc4wyals_elfb_dascoli_source_matrix_20260911-%A_%a.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
runs_root=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
interface=$data_root/tang_themoth_qc4wyals_nested_oof_20260908/interface
source_npz=$interface/qc4wyals_oof_predicted_train_valtail_ada002.npz
selected_checkpoint=${SELECTED_CHECKPOINT:?Set SELECTED_CHECKPOINT to the trained or baseline best.pt}
selected_tag=${SELECTED_TAG:?Set SELECTED_TAG to a short filesystem-safe model name}
lora_rank=${ELF_LORA_RANK:-0}
lora_alpha=${ELF_LORA_ALPHA:-16}
lora_last_n=${ELF_LORA_LAST_N_BLOCKS:-4}
semantic_scale=${DASCOLI_SEMANTIC_SCALE:-1.0}
confidence_floor=${DASCOLI_CONFIDENCE_FLOOR:-0.0}

accuracy=0.50
mode=joint_weighted
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) accuracy=0.00; mode=joint_weighted ;;
  1) accuracy=0.25; mode=joint_weighted ;;
  2) accuracy=0.50; mode=joint_weighted ;;
  3) accuracy=0.75; mode=joint_weighted ;;
  4) accuracy=0.90; mode=joint_weighted ;;
  5) accuracy=1.00; mode=joint_weighted ;;
  6) accuracy=0.50; mode=sentence_only ;;
  7) accuracy=0.50; mode=joint_unweighted ;;
  8) accuracy=0.50; mode=joint_shuffled ;;
  9) accuracy=0.50; mode=semantic_only ;;
  *) exit 2 ;;
esac

run_tag=qc4wyals_elfb_dascoli_eval_${selected_tag}_${mode}_acc${accuracy}_seed49_20260911
output_dir=$runs_root/$run_tag
for required in "$python_bin" "$source_npz" "$selected_checkpoint"; do
  [[ -f "$required" ]] || { echo "Missing required path: $required" >&2; exit 2; }
done
mkdir -p "$output_dir" "$data_root/tmp"
export HF_HOME=$data_root/huggingface TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=$data_root/tmp

cd "$project_root"
args=(
  scripts/train_npz_semantic_to_elf.py
  --npz-path "$source_npz" --output_dir "$output_dir"
  --input-key input_embeddings --sentence-key sentence --model ELF-B
  --checkpoint_path embedded-language-flows/ELF-B-owt-torch
  --init-e2e-checkpoint "$selected_checkpoint" --semantic-adapter-kind flat
  --semantic-hidden-dim 4096 --context_length 64 --freeze-semantic-adapter
  --dascoli-simulated-source --dascoli-train-accuracies 0.25 0.50 0.75 0.90
  --dascoli-eval-accuracy "$accuracy" --dascoli-eval-mode "$mode"
  --dascoli-eval-semantic-scale "$semantic_scale"
  --dascoli-confidence-floor "$confidence_floor"
  --val-num-examples 110 --eval-num-examples 110 --retrieval-num-examples -1
  --batch_size 64 --num_sampling_steps 32 --target-latents-mode lazy
  --target-latent-encode-batch-size 128 --train-target-mask-mode full
  --generation-t5-retrieval --eval-only --checkpoint-metric word_content_mean --seed 49
)
if [[ "$lora_rank" -gt 0 ]]; then
  args+=(
    --elf-lora-rank "$lora_rank" --elf-lora-alpha "$lora_alpha"
    --elf-lora-dropout 0.05 --elf-lora-targets attention,mlp
    --elf-lora-last-n-blocks "$lora_last_n"
  )
fi
exec "$python_bin" -u "${args[@]}"
