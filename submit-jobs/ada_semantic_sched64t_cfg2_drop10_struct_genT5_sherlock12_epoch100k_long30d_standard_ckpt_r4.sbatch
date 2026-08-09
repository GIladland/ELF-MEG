#!/usr/bin/env bash
#SBATCH --job-name=ada_semantic_sched64t_cfg2_drop10_sherlock12_ckpt_r4
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=30-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/ada_semantic_sched64t_cfg2_drop10_struct_genT5_sherlock12_epoch100k_long30d_standard_ckpt_r4-%j.out
#SBATCH --error=submit-jobs/ada_semantic_sched64t_cfg2_drop10_struct_genT5_sherlock12_epoch100k_long30d_standard_ckpt_r4-%j.err

set -euo pipefail

export LIBRIBRAIN_DATA_PATH=/data/engs-pnpl/glandau/MEG2SEM/LibriBrain
export LIBRIBRAIN_SEMANTIC_DATA_PATH=/data/engs-pnpl/glandau/MEG2SEM/LibriBrain
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/ada_semantic_sched64t_cfg2_drop10_struct_genT5_sherlock12_epoch100k_long30d_standard_ckpt_r4
export SKIP_ARC_SETUP=1
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=ada_semantic_sherlock12_eval
export WANDB_DIR=/data/engs-pnpl/glandau/elf-cache/wandb
export WANDB_DATA_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-data
export WANDB_CACHE_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-cache
export TMPDIR=/data/engs-pnpl/glandau/elf-cache/tmp
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"

./scripts/arc_run_meg_context_overfit.sh \
  --condition-source semantic \
  --embedding-type ADA \
  --books 1 2 3 4 5 6 7 8 9 \
  --num-examples 0 \
  --eval-split sherlock1_sessions_11_12 \
  --eval-num-examples 64 \
  --epochs 100000 \
  --batch_size 64 \
  --eval_every 250 \
  --target-encode-batch-size 128 \
  --generation-t5-retrieval \
  --no-wandb-samples \
  --denoiser_loss_weight 1.0 \
  --decoder_loss_weight 1.0 \
  --num_sampling_steps 64 \
  --train-timestep-mode sampling_schedule \
  --train-timestep-steps 64 \
  --cfg_scale 2.0 \
  --self_cond_cfg_scale 1.0 \
  --cond_dropout_prob 0.10 \
  --save-eval-checkpoints \
  --eval-checkpoint-top-k 3 \
  --wandb_run_name ada_semantic_sched64t_cfg2_drop10_struct_genT5_sherlock12_epoch100k_long30d_standard_ckpt_r4 \
  --wandb_notes "Rerun of held-out Sherlock1 sessions 11/12 validation after fixing sessions 11/12 run-2 selection and RoPE generation eval length slicing. Saves eval JSONs and top checkpoints by structured-rank, well_structured_sentence, and mean_rank."
