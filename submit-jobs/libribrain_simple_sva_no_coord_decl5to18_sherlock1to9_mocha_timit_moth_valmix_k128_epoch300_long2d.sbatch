#!/usr/bin/env bash
#SBATCH --job-name=lb_sva_k128
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=submit-jobs/libribrain_simple_sva_no_coord_decl5to18_sherlock1to9_mocha_timit_moth_valmix_k128_epoch300_long2d-%j.out
#SBATCH --error=submit-jobs/libribrain_simple_sva_no_coord_decl5to18_sherlock1to9_mocha_timit_moth_valmix_k128_epoch300_long2d-%j.err

set -euo pipefail

export SKIP_ARC_SETUP=${SKIP_ARC_SETUP:-1}
export FILTER_MODE=${FILTER_MODE:-simple_sv_no_coord}
export FILTER_MIN_WORDS=${FILTER_MIN_WORDS:-5}
export FILTER_MAX_WORDS=${FILTER_MAX_WORDS:-18}
export VAL_INCLUDE_REGEX=${VAL_INCLUDE_REGEX:-'(/Sherlock1/.*_ses-(11|12)_|/TIMIT/.*_ses-14_|/MOCHATIMIT/.*_ses-4_|/TheMoth/.*_ses-29_)'}
export RUN_TAG=${RUN_TAG:-libribrain_simple_sva_no_coord_decl5to18_sherlock1to9_mocha_timit_moth_valmix}
export WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
export WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-conditioning_minilm_simple_sva}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-${RUN_TAG}_k128_epoch300_long2d}
export WANDB_DIR=${WANDB_DIR:-/data/engs-pnpl/glandau/elf-cache/wandb}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-/data/engs-pnpl/glandau/elf-cache/wandb-data}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/data/engs-pnpl/glandau/elf-cache/wandb-cache}
export TMPDIR=${TMPDIR:-/data/engs-pnpl/glandau/elf-cache/tmp}

bash ./scripts/arc_run_event_simple_sva.sh \
  --retrieval-eval-every 2000 \
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
  --wandb_notes "MiniLM pooled K=128. Strict declarative simple-SV/no-coordination 5-18 sentence filter over Sherlock1-9, MOCHA-TIMIT, TIMIT, and TheMoth. Held-out validation sessions: Sherlock1 11/12, TIMIT 14, MOCHATIMIT 4, TheMoth 29."
