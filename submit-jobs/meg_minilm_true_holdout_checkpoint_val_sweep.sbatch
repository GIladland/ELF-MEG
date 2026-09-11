#!/usr/bin/env bash
#SBATCH --job-name=meg_mini_ckpt_val
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-2
#SBATCH --output=submit-jobs/meg_minilm_true_holdout_checkpoint_val_sweep-%A_%a.out
#SBATCH --error=submit-jobs/meg_minilm_true_holdout_checkpoint_val_sweep-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
checkpoint_dir=/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/dt8z7o55/checkpoints

case "$SLURM_ARRAY_TASK_ID" in
  0)
    epoch=852
    val_loss=26.45
    selection=wandb_val_ndcg_priority
    ;;
  1)
    epoch=502
    val_loss=24.44
    selection=wandb_val_ndcg_early
    ;;
  2)
    epoch=843
    val_loss=26.21
    selection=wandb_val_cosine
    ;;
  *) echo "Unsupported task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

export HOLDOUT_SPLIT=val
export DECODE_VARIANT=0
export MEG2SEM_CHECKPOINT=$checkpoint_dir/sherlock_sva_minilm_books1to9_to_sherlock12_bs60_seed49-epoch=${epoch}-val_loss=${val_loss}.ckpt
export RUN_TAG=meg_minilm_TRUE_BRAIN_HOLDOUT_val_CKPTSWEEP_${selection}_epoch${epoch}_s64_cfg0.5_seed7_n48_20260829

exec bash "$PROJECT_ROOT/submit-jobs/meg_minilm_true_holdout_eval.sbatch"
