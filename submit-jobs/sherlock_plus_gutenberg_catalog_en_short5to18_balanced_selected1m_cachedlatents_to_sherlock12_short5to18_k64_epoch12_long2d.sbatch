#!/usr/bin/env bash
#SBATCH --job-name=sh_gutcat1m_k64
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --output=submit-jobs/sherlock_plus_gutenberg_catalog_en_short5to18_balanced_selected1m_cachedlatents_to_sherlock12_short5to18_k64_epoch12_long2d-%j.out
#SBATCH --error=submit-jobs/sherlock_plus_gutenberg_catalog_en_short5to18_balanced_selected1m_cachedlatents_to_sherlock12_short5to18_k64_epoch12_long2d-%j.err

set -euo pipefail

export MODE=${MODE:-gutenberg_plus_sherlock}
export GUTENBERG_CATALOG_CSV_URL=${GUTENBERG_CATALOG_CSV_URL:-https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz}
export GUTENBERG_SHUFFLE_BOOKS=${GUTENBERG_SHUFFLE_BOOKS:-1}
export GUTENBERG_MAX_BOOKS=${GUTENBERG_MAX_BOOKS:-0}
export GUTENBERG_LIMIT_SENTENCES=${GUTENBERG_LIMIT_SENTENCES:-3000000}
export GUTENBERG_QUERY_SAMPLE_SIZE=${GUTENBERG_QUERY_SAMPLE_SIZE:-0}
export GUTENBERG_SELECTION_MODE=${GUTENBERG_SELECTION_MODE:-query_balanced}
export GUTENBERG_PER_QUERY_CANDIDATES=${GUTENBERG_PER_QUERY_CANDIDATES:-80}
export GUTENBERG_SELECTED_COUNT=${GUTENBERG_SELECTED_COUNT:-1000000}
export GUTENBERG_CANDIDATE_FULL_OUTPUT=${GUTENBERG_CANDIDATE_FULL_OUTPUT:-/data/engs-pnpl/glandau/elf-cache/gutenberg/gutenberg_catalog_en_short5to18_raw3m_minilm_seed7.npz}
export GUTENBERG_CANDIDATE_SHORT_OUTPUT=${GUTENBERG_CANDIDATE_SHORT_OUTPUT:-/data/engs-pnpl/glandau/elf-cache/gutenberg/gutenberg_catalog_en_short5to18_content3m_minilm_seed7.npz}
export GUTENBERG_SELECTED_OUTPUT=${GUTENBERG_SELECTED_OUTPUT:-/data/engs-pnpl/glandau/elf-cache/gutenberg/gutenberg_catalog_en_short5to18_querybalanced_selected1m_by_sherlock1to9_short5to18_minilm_seed7.npz}
export COMBINED_OUTPUT=${COMBINED_OUTPUT:-/data/engs-pnpl/glandau/elf-cache/gutenberg/sherlock_plus_gutenberg_catalog_en_short5to18_querybalanced_selected1m_train_sherlock12_short5to18_val_minilm_seed7.npz}
export TARGET_LATENTS_CACHE=${TARGET_LATENTS_CACHE:-/data/engs-pnpl/glandau/elf-cache/gutenberg/sherlock_plus_gutenberg_catalog_en_short5to18_querybalanced_selected1m_train_sherlock12_short5to18_val_t5small_latents_fp16.npy}
export OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/sherlock_plus_gutenberg_catalog_en_short5to18_querybalanced_selected1m_cachedlatents_to_sherlock12_short5to18_k64_epoch12_long2d}
export SKIP_ARC_SETUP=${SKIP_ARC_SETUP:-1}
export WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
export WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-conditioning_minilm_short_holdout}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-sherlock_plus_gutenberg_catalog_en_short5to18_querybalanced_selected1m_cachedlatents_to_sherlock12_short5to18_k64_epoch12_long2d}
export WANDB_DIR=${WANDB_DIR:-/data/engs-pnpl/glandau/elf-cache/wandb}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-/data/engs-pnpl/glandau/elf-cache/wandb-data}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/data/engs-pnpl/glandau/elf-cache/wandb-cache}
export TMPDIR=${TMPDIR:-/data/engs-pnpl/glandau/elf-cache/tmp}

bash ./scripts/arc_run_minilm_short_holdout.sh \
  --target-latents-mode cache \
  --target-latents-cache "$TARGET_LATENTS_CACHE" \
  --target-latents-cache-dtype float16 \
  --target-latent-cache-chunk-size 2048 \
  --target-latent-encode-batch-size 64 \
  --retrieval-eval-every 8000 \
  --retrieval-batch-size 256 \
  --epochs 12 \
  --batch_size 64 \
  --eval_every 4000 \
  --num_sampling_steps 32 \
  --train-timestep-mode sampling_schedule \
  --train-timestep-steps 32 \
  --context_length 64 \
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
  --wandb_notes "Best proven MiniLM pooled bridge: K=64, disk-backed fp16 T5 target-latent cache. Train on Sherlock short 5-18 plus broad Project Gutenberg English short 5-18 query-balanced selection; validate on Sherlock1 sessions 11/12 short."
