#!/usr/bin/env bash
#SBATCH --job-name=elfm_tmpctx
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=submit-jobs/train_qc4wyals_elfm_temporal_context_distill_20260906-%j.out
#SBATCH --error=submit-jobs/train_qc4wyals_elfm_temporal_context_distill_20260906-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
packed=$data_root/tang_themoth_apples_exact_ada002_textmid_seg10_v1/packed
train_npz=$packed/tang_themoth_exact384_ada002_textmid_seg10000_train_dedup_fp16.npz
val_npz=$packed/tang_themoth_exact384_ada002_textmid_seg10000_val_fp16.npz
oracle=/data/engs-pnpl/glandau/elf-runs/elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906/best.pt
run_tag=meg_elfm_temporal64_ctxdistill_exactteacher_ep100_seed49_20260906
output_dir=/data/engs-pnpl/glandau/elf-runs/$run_tag
mkdir -p "$output_dir" "$data_root/tmp" "$data_root/wandb" "$data_root/wandb-data"
export WANDB_ENTITY=giladland-university-of-oxford WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=meg_elfm_temporal_context_20260906 WANDB_RUN_NAME=$run_tag
export WANDB_DIR=$data_root/wandb HF_HOME=$data_root/huggingface TOKENIZERS_PARALLELISM=false
export WANDB_DATA_DIR=$data_root/wandb-data
export PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=$data_root/tmp
cd "$project_root"
exec "$python_bin" scripts/train_npz_meg_to_frozen_elf.py \
  --train-npz "$train_npz" --val-npz "$val_npz" --output_dir "$output_dir" \
  --checkpoint_path "$oracle" --elf-checkpoint-path "$oracle" \
  --semantic-projector-checkpoint "$oracle" --model ELF-M --adapter-kind meg_context \
  --meg-standardization none --meg-clip-boundary -1 --teacher-context-loss-weight 1.0 \
  --checkpoint-metric teacher_context_cosine --device cuda --num-examples 0 \
  --eval-num-examples 110 --retrieval-num-examples 110 --batch_size 8 --epochs 100 \
  --eval_every 1000 --retrieval-eval-every 1000 --retrieval-batch-size 64 \
  --target-encode-batch-size 64 --num_sampling_steps 32 --train-timestep-mode sampling_schedule \
  --train-timestep-steps 32 --context_length 64 --semantic-hidden-dim 4096 \
  --lr 1e-4 --weight_decay 1e-2 --adapter-dropout 0.05 --attention-heads 8 \
  --generation-t5-retrieval --oracle-semantic-eval --skip-final-checkpoint --seed 49 \
  --use_wandb --wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY" \
  --wandb_group "$WANDB_RUN_GROUP" --wandb_run_name "$WANDB_RUN_NAME" \
  --wandb_notes "Direct temporal MEG-to-64-token ELF-M context distillation. Train2652/val110; no semantic bottleneck; protected test26 absent."
