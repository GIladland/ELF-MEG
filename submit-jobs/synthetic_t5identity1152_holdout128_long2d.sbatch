#!/usr/bin/env bash
#SBATCH --job-name=t5id_hold128
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/synthetic_t5identity1152_holdout128_long2d-%j.out
#SBATCH --error=submit-jobs/synthetic_t5identity1152_holdout128_long2d-%j.err

set -euo pipefail

export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/synthetic_t5identity1152_holdout128_long2d
export SYNTHETIC_T5_OUTPUT=/data/engs-pnpl/glandau/elf-cache/synthetic/synthetic_t5_1152_seed7.npz
export SYNTHETIC_COUNT=1152
export SYNTHETIC_SEED=7
export RUN_FROZEN_DECODE=0
export INPUT_KEY=context
export INPUT_MASK_KEY=context_mask
export BRIDGE_TYPE=identity
export SKIP_ARC_SETUP=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=conditioning_holdout_ladder
export WANDB_RUN_NAME=synthetic_t5identity1152_holdout128_long2d
export WANDB_DIR=/data/engs-pnpl/glandau/elf-cache/wandb
export WANDB_DATA_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-data
export WANDB_CACHE_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-cache
export TMPDIR=/data/engs-pnpl/glandau/elf-cache/tmp
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"

bash ./scripts/arc_run_synthetic_t5_context.sh \
  --device cuda \
  --num-examples 1152 \
  --val-num-examples 128 \
  --eval-num-examples 128 \
  --retrieval-eval-every 1000 \
  --retrieval-num-examples 128 \
  --retrieval-batch-size 128 \
  --steps 16000 \
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
  --wandb_sample_examples 16 \
  --wandb_notes "Held-out full T5 identity upper bound: 1024 train + 128 validation sentences. Tests whether native ordered T5 context generalizes under the ELF inversion objective."
