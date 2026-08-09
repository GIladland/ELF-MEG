#!/usr/bin/env bash
#SBATCH --job-name=lb_gut_sva100_k64
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --output=submit-jobs/libribrain_sva_plus_gutenberg_svamatch100k_k64_epoch100_long2d-%j.out
#SBATCH --error=submit-jobs/libribrain_sva_plus_gutenberg_svamatch100k_k64_epoch100_long2d-%j.err

set -euo pipefail

RUN_TAG=${RUN_TAG:-libribrain_simple_sva_no_coord_decl5to18_sherlock1to9_mocha_timit_moth_valmix_plus_gutenberg_svamatch100k}
export NPZ_PATH=${NPZ_PATH:-/data/engs-pnpl/glandau/elf-cache/libribrain_simple/${RUN_TAG}_combined_minilm_seed7.npz}
export OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/${RUN_TAG}_k64_epoch100_long2d}
export TARGET_LATENTS_CACHE=${TARGET_LATENTS_CACHE:-/data/engs-pnpl/glandau/elf-cache/libribrain_simple/${RUN_TAG}_combined_t5small_latents_fp16.npy}
export SKIP_ARC_SETUP=${SKIP_ARC_SETUP:-1}
export WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
export WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-conditioning_minilm_simple_sva}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-${RUN_TAG}_k64_epoch100_long2d}
export WANDB_DIR=${WANDB_DIR:-/data/engs-pnpl/glandau/elf-cache/wandb}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-/data/engs-pnpl/glandau/elf-cache/wandb-data}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/data/engs-pnpl/glandau/elf-cache/wandb-cache}
export TMPDIR=${TMPDIR:-/data/engs-pnpl/glandau/elf-cache/tmp}

bash ./scripts/arc_run_npz_semantic_to_elf.sh \
  --device cuda \
  --num-examples 0 \
  --val-num-examples 322 \
  --eval-num-examples 322 \
  --retrieval-num-examples 322 \
  --retrieval-eval-every 4000 \
  --retrieval-batch-size 256 \
  --epochs 100 \
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
  --target-latents-mode cache \
  --target-latents-cache "$TARGET_LATENTS_CACHE" \
  --target-latents-cache-dtype float16 \
  --target-latent-cache-chunk-size 4096 \
  --target-latent-encode-batch-size 128 \
  --generation-t5-retrieval \
  --wandb_sample_examples 16 \
  --save-best-checkpoint \
  --seed 7 \
  --wandb_notes "MiniLM pooled K=64. Current LibriBrain/Sherlock SVA train plus 100k Gutenberg short SVA-like sentences selected query-balanced by MiniLM similarity to SVA train. Held-out SVA validation sessions unchanged."
