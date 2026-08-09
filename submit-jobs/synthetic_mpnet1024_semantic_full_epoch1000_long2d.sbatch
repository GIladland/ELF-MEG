#!/usr/bin/env bash
#SBATCH --job-name=mpnet_pool_e1000
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/synthetic_mpnet1024_semantic_full_epoch1000_long2d-%j.out
#SBATCH --error=submit-jobs/synthetic_mpnet1024_semantic_full_epoch1000_long2d-%j.err

set -euo pipefail

export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/synthetic_mpnet1024_semantic_full_epoch1000_long2d
export SYNTHETIC_COUNT=1024
export SYNTHETIC_SEED=7
export SYNTHETIC_OUTPUT=/data/engs-pnpl/glandau/elf-cache/synthetic/synthetic_mpnet_1024_seed7.npz
export SYNTHETIC_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
export SYNTHETIC_BATCH_SIZE=128
export SKIP_ARC_SETUP=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=conditioning_ladder
export WANDB_RUN_NAME=synthetic_mpnet1024_semantic_full_epoch1000_long2d
export WANDB_DIR=/data/engs-pnpl/glandau/elf-cache/wandb
export WANDB_DATA_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-data
export WANDB_CACHE_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-cache
export TMPDIR=/data/engs-pnpl/glandau/elf-cache/tmp
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"

bash ./scripts/arc_run_npz_semantic_to_elf.sh \
  --device cuda \
  --num-examples 1024 \
  --eval-num-examples 128 \
  --retrieval-eval-every 1000 \
  --retrieval-num-examples 1024 \
  --retrieval-batch-size 512 \
  --steps 16000 \
  --batch_size 64 \
  --eval_every 500 \
  --num_sampling_steps 32 \
  --train-timestep-mode sampling_schedule \
  --train-timestep-steps 32 \
  --context_length 20 \
  --semantic-hidden-dim 2048 \
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
  --wandb_notes "Pooled MPNet conditioning run: same 1024 synthetic sentences, K=20 ELF context, and 16000-step schedule as the longer pooled-MiniLM baseline."
