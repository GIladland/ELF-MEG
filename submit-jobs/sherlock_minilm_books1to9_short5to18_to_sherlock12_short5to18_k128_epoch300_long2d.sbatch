#!/usr/bin/env bash
#SBATCH --job-name=sh_short_k128
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/sherlock_minilm_books1to9_short5to18_to_sherlock12_short5to18_k128_epoch300_long2d-%j.out
#SBATCH --error=submit-jobs/sherlock_minilm_books1to9_short5to18_to_sherlock12_short5to18_k128_epoch300_long2d-%j.err

set -euo pipefail

export MODE=${MODE:-sherlock}
export OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/sherlock_minilm_books1to9_short5to18_to_sherlock12_short5to18_k128_epoch300_long2d}
export COMBINED_OUTPUT=${COMBINED_OUTPUT:-/data/engs-pnpl/glandau/elf-cache/sherlock/sherlock_books1to9_short5to18_train_sherlock12_short5to18_val_minilm_seed7.npz}
export SKIP_ARC_SETUP=${SKIP_ARC_SETUP:-1}
export WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
export WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-conditioning_minilm_short_holdout}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-sherlock_minilm_books1to9_short5to18_to_sherlock12_short5to18_k128_epoch300_long2d}
export WANDB_DIR=${WANDB_DIR:-/data/engs-pnpl/glandau/elf-cache/wandb}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-/data/engs-pnpl/glandau/elf-cache/wandb-data}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/data/engs-pnpl/glandau/elf-cache/wandb-cache}
export TMPDIR=${TMPDIR:-/data/engs-pnpl/glandau/elf-cache/tmp}

bash ./scripts/arc_run_minilm_short_holdout.sh \
  --retrieval-eval-every 4000 \
  --retrieval-batch-size 256 \
  --epochs 300 \
  --batch_size 64 \
  --eval_every 1000 \
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
  --wandb_notes "MiniLM pooled K=128. Train on Sherlock books 1-9 filtered to content-bearing 5-18 word sentences. Validate on Sherlock1 sessions 11/12 with the same 5-18 content filter."
