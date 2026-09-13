#!/usr/bin/env bash
#SBATCH --job-name=elfb_dasgrid
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-24%4
#SBATCH --output=submit-jobs/eval_qc4wyals_elfb_dascoli_weight_grid_20260912-%A_%a.out
#SBATCH --error=submit-jobs/eval_qc4wyals_elfb_dascoli_weight_grid_20260912-%A_%a.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
runs_root=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
interface=$data_root/tang_themoth_qc4wyals_nested_oof_20260908/interface
source_npz=$interface/qc4wyals_oof_predicted_train_valtail_ada002.npz
selected_checkpoint=${SELECTED_CHECKPOINT:?Set SELECTED_CHECKPOINT to the trained best.pt}
selected_tag=${SELECTED_TAG:-lora8_mid_ep40}
lora_rank=${ELF_LORA_RANK:-8}
lora_alpha=${ELF_LORA_ALPHA:-16}
lora_last_n=${ELF_LORA_LAST_N_BLOCKS:-4}

semantic_scales=(0.0 0.1 0.25 0.5 1.0)
confidence_floors=(0.0 0.25 0.5 0.75 1.0)
task_id=${SLURM_ARRAY_TASK_ID:-0}
semantic_scale=${semantic_scales[$((task_id / 5))]}
confidence_floor=${confidence_floors[$((task_id % 5))]}
run_tag=qc4wyals_elfb_dascoli_weightgrid_${selected_tag}_sem${semantic_scale}_floor${confidence_floor}_acc0.50_seed49_20260912
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
  --dascoli-eval-accuracy 0.50 --dascoli-eval-mode joint_weighted
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
