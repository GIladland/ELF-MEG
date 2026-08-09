#!/usr/bin/env bash
#SBATCH --job-name=minilm_tok_e1000
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/synthetic_minilm_tokens1024_t5bridge20_epoch1000_long2d-%j.out
#SBATCH --error=submit-jobs/synthetic_minilm_tokens1024_t5bridge20_epoch1000_long2d-%j.err

set -euo pipefail

export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/synthetic_minilm_tokens1024_t5bridge20_epoch1000_long2d
export SYNTHETIC_OUTPUT=/data/engs-pnpl/glandau/elf-cache/synthetic/synthetic_minilm_tokens_1024_seed7.npz
export SYNTHETIC_COUNT=1024
export SYNTHETIC_SEED=7
export SYNTHETIC_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export SYNTHETIC_SOURCE_MAX_LENGTH=32
export SYNTHETIC_BATCH_SIZE=256
export BRIDGE_TYPE=resampler
export SKIP_ARC_SETUP=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=conditioning_ladder
export WANDB_RUN_NAME=synthetic_minilm_tokens1024_t5bridge20_epoch1000_long2d
export WANDB_DIR=/data/engs-pnpl/glandau/elf-cache/wandb
export WANDB_DATA_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-data
export WANDB_CACHE_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-cache
export TMPDIR=/data/engs-pnpl/glandau/elf-cache/tmp
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"

bash ./scripts/arc_run_token_embedding_t5_bridge.sh \
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
  --bridge-layers 2 \
  --bridge-heads 8 \
  --cfg_scale 1.0 \
  --self_cond_cfg_scale 1.0 \
  --cond_dropout_prob 0.0 \
  --lr 1e-4 \
  --denoiser_loss_weight 1.0 \
  --decoder_loss_weight 1.0 \
  --wandb_sample_examples 16 \
  --wandb_notes "Longer token-MiniLM conditioning run: same token hidden-state source, learned K=20 resampler, and 1024 synthetic sentences as the earlier token run, but 16000 steps / 1000 epochs."
