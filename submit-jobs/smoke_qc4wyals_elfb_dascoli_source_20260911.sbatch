#!/usr/bin/env bash
#SBATCH --job-name=elfb_dassmoke
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --oversubscribe
#SBATCH --output=submit-jobs/smoke_qc4wyals_elfb_dascoli_source_20260911-%j.out
#SBATCH --error=submit-jobs/smoke_qc4wyals_elfb_dascoli_source_20260911-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
runs_root=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
interface=$data_root/tang_themoth_qc4wyals_nested_oof_20260908/interface
source_npz=$interface/qc4wyals_oof_predicted_train_valtail_ada002.npz
old_x1_best=$runs_root/elf_semantic_scaling_meg_ada_x1_trained_adapter_val110_seed49_20260902/best.pt
output_dir=$runs_root/qc4wyals_elfb_dascoli_source_smoke_${SLURM_JOB_ID}

for required in "$python_bin" "$source_npz" "$old_x1_best"; do
  [[ -f "$required" ]] || { echo "Missing required path: $required" >&2; exit 2; }
done
mkdir -p "$output_dir" "$data_root/tmp"
export HF_HOME=$data_root/huggingface TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=$data_root/tmp

cd "$project_root"
exec "$python_bin" -u scripts/train_npz_semantic_to_elf.py \
  --npz-path "$source_npz" --output_dir "$output_dir" \
  --input-key input_embeddings --sentence-key sentence --model ELF-B \
  --checkpoint_path embedded-language-flows/ELF-B-owt-torch \
  --init-e2e-checkpoint "$old_x1_best" --semantic-adapter-kind flat \
  --semantic-hidden-dim 4096 --context_length 64 --freeze-semantic-adapter \
  --elf-lora-rank 4 --elf-lora-alpha 8 --elf-lora-dropout 0.05 \
  --elf-lora-targets attention,mlp --elf-lora-last-n-blocks 4 --elf-lr 1e-5 \
  --dascoli-simulated-source --dascoli-train-accuracies 0.25 0.50 0.75 0.90 \
  --dascoli-eval-accuracy 0.50 --dascoli-eval-mode joint_weighted \
  --dascoli-source-dropout-prob 0.10 --num-examples 8 --val-num-examples 1 \
  --eval-num-examples 1 --retrieval-num-examples -1 --batch_size 2 --steps 1 \
  --eval_every 1 --num_sampling_steps 2 --train-timestep-mode sampling_schedule \
  --train-timestep-steps 2 --target-latents-mode precompute \
  --target-latent-encode-batch-size 8 --train-target-mask-mode full \
  --cond_dropout_prob 0.10 --denoiser_loss_weight 1.0 --decoder_loss_weight 1.0 \
  --checkpoint-metric word_content_mean --save-best-checkpoint --seed 49
