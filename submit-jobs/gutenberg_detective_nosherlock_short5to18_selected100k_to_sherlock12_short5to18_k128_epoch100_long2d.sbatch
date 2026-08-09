#!/usr/bin/env bash
#SBATCH --job-name=gut_short_k128
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --output=submit-jobs/gutenberg_detective_nosherlock_short5to18_selected100k_to_sherlock12_short5to18_k128_epoch100_long2d-%j.out
#SBATCH --error=submit-jobs/gutenberg_detective_nosherlock_short5to18_selected100k_to_sherlock12_short5to18_k128_epoch100_long2d-%j.err

set -euo pipefail

export MODE=${MODE:-gutenberg}
export OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/gutenberg_detective_nosherlock_short5to18_selected100k_to_sherlock12_short5to18_k128_epoch100_long2d}
export COMBINED_OUTPUT=${COMBINED_OUTPUT:-/data/engs-pnpl/glandau/elf-cache/gutenberg/gutenberg_detective_nosherlock_short5to18_selected100k_train_sherlock12_short5to18_val_minilm_seed7.npz}
export GUTENBERG_SELECTED_OUTPUT=${GUTENBERG_SELECTED_OUTPUT:-/data/engs-pnpl/glandau/elf-cache/gutenberg/gutenberg_detective_nosherlock_short5to18_selected100k_by_sherlock1to9_short5to18_minilm_seed7.npz}
export GUTENBERG_SELECTED_COUNT=${GUTENBERG_SELECTED_COUNT:-100000}
export SKIP_ARC_SETUP=${SKIP_ARC_SETUP:-1}
export WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
export WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-conditioning_minilm_short_holdout}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-gutenberg_detective_nosherlock_short5to18_selected100k_to_sherlock12_short5to18_k128_epoch100_long2d}
export WANDB_DIR=${WANDB_DIR:-/data/engs-pnpl/glandau/elf-cache/wandb}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-/data/engs-pnpl/glandau/elf-cache/wandb-data}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/data/engs-pnpl/glandau/elf-cache/wandb-cache}
export TMPDIR=${TMPDIR:-/data/engs-pnpl/glandau/elf-cache/tmp}

bash ./scripts/arc_run_minilm_short_holdout.sh \
  --retrieval-eval-every 8000 \
  --retrieval-batch-size 256 \
  --epochs 100 \
  --batch_size 64 \
  --eval_every 4000 \
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
  --wandb_notes "MiniLM pooled K=128. Filter clean Gutenberg detective/mystery candidates to content-bearing 5-18 word sentences, select the top 100k nearest to Sherlock books 1-9 short sentences, and validate on Sherlock1 sessions 11/12 short sentences."
