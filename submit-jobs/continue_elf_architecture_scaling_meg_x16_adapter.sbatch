#!/usr/bin/env bash
#SBATCH --job-name=elf_x16_acont
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-1%2
#SBATCH --output=submit-jobs/continue_elf_architecture_scaling_meg_x16_adapter-%A_%a.out
#SBATCH --error=submit-jobs/continue_elf_architecture_scaling_meg_x16_adapter-%A_%a.err

set -euo pipefail

# Continue from the best checkpoint retained by the interrupted x16 adapter
# runs.  The parent jobs stopped only because the shared filesystem filled.
# Avoid per-evaluation checkpoint snapshots: best.pt is sufficient and keeps
# the two large-model runs below roughly 5 GB instead of 170+ GB.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) model=ELF-M; model_slug=elfm; export LADDER_TASK_ID_OVERRIDE=5 ;;
  1) model=ELF-L; model_slug=elfl; export LADDER_TASK_ID_OVERRIDE=5 ;;
  *) exit 2 ;;
esac

export MODEL_OVERRIDE=$model
export CHECKPOINT_PATH_OVERRIDE=embedded-language-flows/${model}-owt-torch
export ORACLE_TAG_OVERRIDE=elf_archscale_meg_ada_x16_${model_slug}_trained_adapter_val110_seed49_20260904
export RUN_TAG_OVERRIDE=elf_archscale_meg_ada_x16_${model_slug}_trained_adapter_continue1_val110_seed49_20260905
export RETRIEVAL_NUM_EXAMPLES_OVERRIDE=-1
export BATCH_SIZE_OVERRIDE=16
export SAVE_FINAL_CHECKPOINT_OVERRIDE=0
export SAVE_EVAL_CHECKPOINTS_OVERRIDE=0
export EVAL_CHECKPOINT_TOP_K_OVERRIDE=1
export EPOCHS_OVERRIDE=60

exec bash "$PROJECT_ROOT/submit-jobs/train_elf_semantic_scaling_adapter_ladder.sbatch"
