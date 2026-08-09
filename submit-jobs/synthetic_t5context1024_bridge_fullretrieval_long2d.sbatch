#!/usr/bin/env bash
#SBATCH --job-name=synthetic_t5ctx1024
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/synthetic_t5context1024_bridge_fullretrieval_long2d-%j.out
#SBATCH --error=submit-jobs/synthetic_t5context1024_bridge_fullretrieval_long2d-%j.err

set -euo pipefail

export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/synthetic_t5context1024_bridge_fullretrieval_long2d
export SYNTHETIC_T5_OUTPUT=/data/engs-pnpl/glandau/elf-cache/synthetic/synthetic_t5_1024_seed7.npz
export SYNTHETIC_COUNT=1024
export SYNTHETIC_SEED=7
export RUN_FROZEN_DECODE=1
export FROZEN_DECODE_EXAMPLES=128
export FROZEN_DECODE_STEPS=32
export FROZEN_RETRIEVAL_EXAMPLES=1024
export FROZEN_RETRIEVAL_BATCH_SIZE=512
export SKIP_ARC_SETUP=1
export WANDB_RUN_GROUP=conditioning_ladder
export WANDB_RUN_NAME=synthetic_t5context1024_bridge_fullretrieval_long2d
export WANDB_DIR=/data/engs-pnpl/glandau/elf-cache/wandb
export WANDB_DATA_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-data
export WANDB_CACHE_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-cache
export TMPDIR=/data/engs-pnpl/glandau/elf-cache/tmp
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"

./scripts/arc_run_synthetic_t5_context.sh \
  --device cuda \
  --num-examples 1024 \
  --eval-num-examples 128 \
  --retrieval-eval-every 1000 \
  --retrieval-num-examples 1024 \
  --retrieval-batch-size 512 \
  --context_length 20 \
  --bridge-layers 2 \
  --bridge-heads 8 \
  --steps 8000 \
  --batch_size 64 \
  --eval_every 500 \
  --num_sampling_steps 32 \
  --train-timestep-mode sampling_schedule \
  --train-timestep-steps 32 \
  --cfg_scale 1.0 \
  --self_cond_cfg_scale 1.0 \
  --cond_dropout_prob 0.0 \
  --lr 1e-4 \
  --denoiser_loss_weight 1.0 \
  --decoder_loss_weight 1.0 \
  --wandb_notes "First-principles oracle T5 context run: 1024 simple synthetic sentences, frozen decode baseline, full conditional retrieval, generation coherency, generated-text T5 retrieval."
