#!/usr/bin/env bash
#SBATCH --job-name=elf_x16_exact
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-1%2
#SBATCH --output=submit-jobs/eval_elf_architecture_scaling_meg_x16_exact-%A_%a.out
#SBATCH --error=submit-jobs/eval_elf_architecture_scaling_meg_x16_exact-%A_%a.err

set -euo pipefail
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) model=ELF-M; model_slug=elfm ;;
  1) model=ELF-L; model_slug=elfl ;;
  *) exit 2 ;;
esac
export LADDER_TASK_ID_OVERRIDE=1
export MODEL_OVERRIDE=$model
export CHECKPOINT_PATH_OVERRIDE=embedded-language-flows/${model}-owt-torch
export ORACLE_TAG_OVERRIDE=elf_archscale_meg_ada_x16_${model_slug}_continue_x1_content_seed7_20260904
export RUN_TAG_OVERRIDE=eval_elf_archscale_meg_ada_x16_${model_slug}_exact_val110_seed49_20260904
export BERT_DEVICE=cuda
exec bash "$PROJECT_ROOT/submit-jobs/eval_elf_semantic_scaling_x1_exact.sbatch"
