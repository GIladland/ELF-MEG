#!/usr/bin/env bash
#SBATCH --job-name=tokmini_h128_k20
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/synthetic_minilm_tokens1152_holdout128_k20_epoch1000_long2d-%j.out
#SBATCH --error=submit-jobs/synthetic_minilm_tokens1152_holdout128_k20_epoch1000_long2d-%j.err

set -euo pipefail

export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/synthetic_minilm_tokens1152_holdout128_k20_epoch1000_long2d
export SYNTHETIC_COUNT=1152
export SYNTHETIC_SEED=7
export SYNTHETIC_OUTPUT=/data/engs-pnpl/glandau/elf-cache/synthetic/synthetic_minilm_tokens_1152_seed7.npz
export SYNTHETIC_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export SYNTHETIC_SOURCE_MAX_LENGTH=32
export SYNTHETIC_BATCH_SIZE=256
export BRIDGE_TYPE=resampler
export SKIP_ARC_SETUP=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=conditioning_holdout_ladder
export WANDB_RUN_NAME=synthetic_minilm_tokens1152_holdout128_k20_epoch1000_long2d
export WANDB_DIR=/data/engs-pnpl/glandau/elf-cache/wandb
export WANDB_DATA_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-data
export WANDB_CACHE_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-cache
export TMPDIR=/data/engs-pnpl/glandau/elf-cache/tmp
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"

bash ./scripts/arc_run_token_embedding_t5_bridge.sh \
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
  --context_length 20 \
  --bridge-layers 2 \
  --bridge-heads 8 \
  --cfg_scale 1.0 \
  --self_cond_cfg_scale 1.0 \
  --cond_dropout_prob 0.0 \
  --lr 1e-4 \
  --denoiser_loss_weight 1.0 \
  --decoder_loss_weight 1.0 \
  --wandb_sample_examples 16 \
  --seed 7 \
  --wandb_notes "Held-out token-level MiniLM control: 1024 train + 128 validation, MiniLM token hidden states resampled to K=20 ELF context tokens."
