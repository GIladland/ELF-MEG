#!/usr/bin/env bash
#SBATCH --job-name=t5order20_1024
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/synthetic_t5_orderpreserve20_1024_full_long2d-%j.out
#SBATCH --error=submit-jobs/synthetic_t5_orderpreserve20_1024_full_long2d-%j.err

set -euo pipefail

export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/synthetic_t5_orderpreserve20_1024_full_long2d
export SYNTHETIC_T5_OUTPUT=/data/engs-pnpl/glandau/elf-cache/synthetic/synthetic_t5_1024_seed7.npz
export SYNTHETIC_COUNT=1024
export SYNTHETIC_SEED=7
export RUN_FROZEN_DECODE=0
export INPUT_KEY=target_t5_latents
export BRIDGE_TYPE=order_preserving
export SKIP_ARC_SETUP=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=conditioning_ladder
export WANDB_RUN_NAME=synthetic_t5_orderpreserve20_1024_full_long2d
export WANDB_DIR=/data/engs-pnpl/glandau/elf-cache/wandb
export WANDB_DATA_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-data
export WANDB_CACHE_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-cache
export TMPDIR=/data/engs-pnpl/glandau/elf-cache/tmp
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"

bash ./scripts/arc_run_synthetic_t5_context.sh \
  --device cuda \
  --num-examples 1024 \
  --eval-num-examples 128 \
  --retrieval-eval-every 1000 \
  --retrieval-num-examples 1024 \
  --retrieval-batch-size 512 \
  --steps 8000 \
  --batch_size 64 \
  --eval_every 500 \
  --num_sampling_steps 32 \
  --train-timestep-mode sampling_schedule \
  --train-timestep-steps 32 \
  --context_length 20 \
  --bridge-dropout 0.0 \
  --cfg_scale 1.0 \
  --self_cond_cfg_scale 1.0 \
  --cond_dropout_prob 0.0 \
  --lr 1e-4 \
  --denoiser_loss_weight 1.0 \
  --decoder_loss_weight 1.0 \
  --wandb_sample_examples 16 \
  --wandb_notes "Order-preserving T5 bridge diagnostic: stored T5 context latents and masks are selected by index, then passed through an identity-initialized per-token residual adapter. Preserves token order and K=20 bandwidth to isolate whether cross-attention resampling damaged the native T5 condition."
