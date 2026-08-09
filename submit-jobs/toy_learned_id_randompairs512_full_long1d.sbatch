#!/usr/bin/env bash
#SBATCH --job-name=toy_learned_id_512_full
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/toy_learned_id_randompairs512_full_long1d-%j.out
#SBATCH --error=submit-jobs/toy_learned_id_randompairs512_full_long1d-%j.err

set -euo pipefail

export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/toy_learned_id_randompairs512_full_long1d
export SKIP_ARC_SETUP=1
export WANDB_RUN_GROUP=conditioning_ladder
export WANDB_RUN_NAME=toy_learned_id_randompairs512_full_long1d
export WANDB_DIR=/data/engs-pnpl/glandau/elf-cache/wandb
export WANDB_DATA_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-data
export WANDB_CACHE_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-cache
export TMPDIR=/data/engs-pnpl/glandau/elf-cache/tmp
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"

./scripts/arc_run_toy_context_overfit.sh \
  --context_mode learned \
  --dataset_mode random_pairs \
  --dataset_size 512 \
  --steps 5000 \
  --batch_size 64 \
  --eval_every 250 \
  --num_sampling_steps 32 \
  --cfg_scale 1.0 \
  --self_cond_cfg_scale 1.0 \
  --lr 1e-4 \
  --last_n_blocks -1 \
  --denoiser_loss_weight 1.0 \
  --decoder_loss_weight 1.0 \
  --wandb_run_name "$WANDB_RUN_NAME" \
  --seed 101
