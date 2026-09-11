#!/usr/bin/env bash
#SBATCH --job-name=elf_x16_adapt
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-1%2
#SBATCH --output=submit-jobs/train_elf_architecture_scaling_meg_x16_adapter-%A_%a.out
#SBATCH --error=submit-jobs/train_elf_architecture_scaling_meg_x16_adapter-%A_%a.err

set -euo pipefail
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) model=ELF-M; model_slug=elfm ;;
  1) model=ELF-L; model_slug=elfl ;;
  *) exit 2 ;;
esac
export LADDER_TASK_ID_OVERRIDE=5
export MODEL_OVERRIDE=$model
export CHECKPOINT_PATH_OVERRIDE=embedded-language-flows/${model}-owt-torch
export ORACLE_TAG_OVERRIDE=elf_archscale_meg_ada_x16_${model_slug}_continue_x1_content_seed7_20260904
export RUN_TAG_OVERRIDE=elf_archscale_meg_ada_x16_${model_slug}_trained_adapter_val110_seed49_20260904
export RETRIEVAL_NUM_EXAMPLES_OVERRIDE=-1
export BATCH_SIZE_OVERRIDE=16
exec bash "$PROJECT_ROOT/submit-jobs/train_elf_semantic_scaling_adapter_ladder.sbatch"
