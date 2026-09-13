#!/usr/bin/env bash
#SBATCH --job-name=elfb_dasflow
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-2%3
#SBATCH --output=submit-jobs/train_qc4wyals_elfb_dascoli_source_matrix_20260911-%A_%a.out
#SBATCH --error=submit-jobs/train_qc4wyals_elfb_dascoli_source_matrix_20260911-%A_%a.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
runs_root=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
interface=$data_root/tang_themoth_qc4wyals_nested_oof_20260908/interface
source_npz=$interface/qc4wyals_oof_predicted_train_valtail_ada002.npz
old_x1_best=$runs_root/elf_semantic_scaling_meg_ada_x1_trained_adapter_val110_seed49_20260902/best.pt

lora_rank=4
lora_alpha=8
lora_targets=attention,mlp
lora_last_n=4
last_n=-1
elf_lr=2e-6
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) arm=lora4_low; elf_lr=2e-6 ;;
  1) arm=lora8_mid; lora_rank=8; lora_alpha=16; elf_lr=1e-5 ;;
  2) arm=full_last2; lora_rank=0; last_n=2; elf_lr=5e-7 ;;
  *) exit 2 ;;
esac

epochs=${DASCOLI_EPOCHS:-40}
run_tag=qc4wyals_elfb_dascoli_${arm}_ep${epochs}_seed49_20260911
output_dir=$runs_root/$run_tag
for required in "$python_bin" "$source_npz" "$old_x1_best"; do
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
  --init-e2e-checkpoint "$old_x1_best" --semantic-adapter-kind flat
  --semantic-hidden-dim 4096 --context_length 64 --freeze-semantic-adapter
  --dascoli-simulated-source --dascoli-train-accuracies 0.25 0.50 0.75 0.90
  --dascoli-eval-accuracy 0.50 --dascoli-eval-mode joint_weighted
  --dascoli-source-dropout-prob 0.10 --val-num-examples 110
  --eval-num-examples 110 --retrieval-num-examples -1 --batch_size 64
  --epochs "$epochs" --eval_every 420 --num_sampling_steps 32
  --train-timestep-mode sampling_schedule --train-timestep-steps 32
  --target-latents-mode precompute --target-latent-encode-batch-size 128
  --train-target-mask-mode full --lr "$elf_lr" --elf-lr "$elf_lr"
  --lr-schedule cosine --warmup-steps 100 --min-lr-ratio 0.05
  --weight_decay 0.0 --cond_dropout_prob 0.10
  --denoiser_loss_weight 1.0 --decoder_loss_weight 1.0
  --decoder-content-token-weight 2.0 --decoder-content-bow-loss-weight 0.10
  --decoder-content-precision-loss-weight 0.05
  --checkpoint-metric word_content_mean --save-best-checkpoint
  --save-trainable-only-checkpoint --seed 49
)
if [[ "$lora_rank" -gt 0 ]]; then
  args+=(
    --elf-lora-rank "$lora_rank" --elf-lora-alpha "$lora_alpha"
    --elf-lora-dropout 0.05 --elf-lora-targets "$lora_targets"
    --elf-lora-last-n-blocks "$lora_last_n"
  )
else
  args+=(--last_n_blocks "$last_n")
fi
exec "$python_bin" -u "${args[@]}"
