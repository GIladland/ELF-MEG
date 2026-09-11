#!/usr/bin/env bash
#SBATCH --job-name=elf_arch_adapt
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-3%2
#SBATCH --output=submit-jobs/train_elf_architecture_scaling_x1_adapter-%A_%a.out
#SBATCH --error=submit-jobs/train_elf_architecture_scaling_x1_adapter-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) modality=mri_minilm; model=ELF-M; model_slug=elfm; audit_n=266; export LADDER_TASK_ID_OVERRIDE=4 ;;
  1) modality=mri_minilm; model=ELF-L; model_slug=elfl; audit_n=266; export LADDER_TASK_ID_OVERRIDE=4 ;;
  2) modality=meg_ada; model=ELF-M; model_slug=elfm; audit_n=110; export LADDER_TASK_ID_OVERRIDE=5 ;;
  3) modality=meg_ada; model=ELF-L; model_slug=elfl; audit_n=110; export LADDER_TASK_ID_OVERRIDE=5 ;;
  *) exit 2 ;;
esac

export MODEL_OVERRIDE=$model
export CHECKPOINT_PATH_OVERRIDE=embedded-language-flows/${model}-owt-torch
export ORACLE_TAG_OVERRIDE=elf_archscale_${modality}_x1_${model_slug}_exact_compute44k_content_seed7_20260904
export RUN_TAG_OVERRIDE=elf_archscale_${modality}_x1_${model_slug}_trained_adapter_val${audit_n}_seed49_20260904
export RETRIEVAL_NUM_EXAMPLES_OVERRIDE=-1
export BATCH_SIZE_OVERRIDE=16

exec bash "$PROJECT_ROOT/submit-jobs/train_elf_semantic_scaling_adapter_ladder.sbatch"
