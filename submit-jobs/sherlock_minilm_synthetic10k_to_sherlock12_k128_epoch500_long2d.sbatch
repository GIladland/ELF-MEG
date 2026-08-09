#!/usr/bin/env bash
#SBATCH --job-name=mini_syn2sh_k128
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/sherlock_minilm_synthetic10k_to_sherlock12_k128_epoch500_long2d-%j.out
#SBATCH --error=submit-jobs/sherlock_minilm_synthetic10k_to_sherlock12_k128_epoch500_long2d-%j.err

set -euo pipefail

export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/sherlock_minilm_synthetic10k_to_sherlock12_k128_epoch500_long2d
export COMBINED_OUTPUT=/data/engs-pnpl/glandau/elf-cache/sherlock/synthetic10k_train_sherlock12_val_minilm_seed7.npz
export SYNTHETIC_OUTPUT=/data/engs-pnpl/glandau/elf-cache/synthetic/synthetic_minilm_11264_seed7.npz
export SYNTHETIC_COUNT=11264
export SYNTHETIC_LIMIT=10240
export INCLUDE_SHERLOCK_TRAIN=0
export SKIP_ARC_SETUP=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=conditioning_minilm_sherlock_holdout
export WANDB_RUN_NAME=sherlock_minilm_synthetic10k_to_sherlock12_k128_epoch500_long2d
export WANDB_DIR=/data/engs-pnpl/glandau/elf-cache/wandb
export WANDB_DATA_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-data
export WANDB_CACHE_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-cache
export TMPDIR=/data/engs-pnpl/glandau/elf-cache/tmp

bash ./scripts/arc_run_minilm_sherlock_holdout.sh \
  --retrieval-eval-every 4000 \
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
  --wandb_notes "OOD corpus test: train MiniLM pooled K=128 on 10240 synthetic template sentences only, validate on held-out Sherlock1 sessions 11/12 sentence rows."
