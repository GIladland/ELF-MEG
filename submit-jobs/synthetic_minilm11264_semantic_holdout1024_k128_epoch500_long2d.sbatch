#!/usr/bin/env bash
#SBATCH --job-name=minilm10k_k128
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/synthetic_minilm11264_semantic_holdout1024_k128_epoch500_long2d-%j.out
#SBATCH --error=submit-jobs/synthetic_minilm11264_semantic_holdout1024_k128_epoch500_long2d-%j.err

set -euo pipefail

export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/synthetic_minilm11264_semantic_holdout1024_k128_epoch500_long2d
export SYNTHETIC_COUNT=11264
export SYNTHETIC_SEED=7
export SYNTHETIC_OUTPUT=/data/engs-pnpl/glandau/elf-cache/synthetic/synthetic_minilm_11264_seed7.npz
export SYNTHETIC_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export SYNTHETIC_POOLING=mean
export SYNTHETIC_BATCH_SIZE=512
export SKIP_ARC_SETUP=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=conditioning_minilm_focus
export WANDB_RUN_NAME=synthetic_minilm11264_semantic_holdout1024_k128_epoch500_long2d
export WANDB_DIR=/data/engs-pnpl/glandau/elf-cache/wandb
export WANDB_DATA_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-data
export WANDB_CACHE_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-cache
export TMPDIR=/data/engs-pnpl/glandau/elf-cache/tmp
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"

bash ./scripts/arc_run_npz_semantic_to_elf.sh \
  --device cuda \
  --num-examples 11264 \
  --val-num-examples 1024 \
  --eval-num-examples 1024 \
  --retrieval-eval-every 4000 \
  --retrieval-num-examples 1024 \
  --retrieval-batch-size 256 \
  --epochs 500 \
  --batch_size 64 \
  --eval_every 2000 \
  --num_sampling_steps 32 \
  --train-timestep-mode sampling_schedule \
  --train-timestep-steps 32 \
  --context_length 128 \
  --semantic-hidden-dim 4096 \
  --last_n_blocks -1 \
  --lr 1e-4 \
  --cfg_scale 1.0 \
  --self_cond_cfg_scale 1.0 \
  --cond_dropout_prob 0.0 \
  --denoiser_loss_weight 1.0 \
  --decoder_loss_weight 1.0 \
  --generation-t5-retrieval \
  --wandb_sample_examples 16 \
  --save-best-checkpoint \
  --seed 7 \
  --wandb_notes "MiniLM pooled K ablation: same 10240 train + 1024 validation data as K=64, but bridge expands one MiniLM vector to K=128 ELF context tokens."
