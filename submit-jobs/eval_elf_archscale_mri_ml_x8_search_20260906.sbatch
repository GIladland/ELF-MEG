#!/usr/bin/env bash
#SBATCH --job-name=elf_mri_ml_x8exact
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-3%2
#SBATCH --output=submit-jobs/eval_elf_archscale_mri_ml_x8_search_20260906-%A_%a.out
#SBATCH --error=submit-jobs/eval_elf_archscale_mri_ml_x8_search_20260906-%A_%a.err

set -euo pipefail
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) model=ELF-M; model_slug=elfm; recipe=x8_const2e5 ;;
  1) model=ELF-M; model_slug=elfm; recipe=x8_cos1e5 ;;
  2) model=ELF-L; model_slug=elfl; recipe=x8_const1e5 ;;
  3) model=ELF-L; model_slug=elfl; recipe=x8_cos5e6 ;;
  *) exit 2 ;;
esac

export LADDER_TASK_ID_OVERRIDE=0
export MODEL_OVERRIDE=$model
export CHECKPOINT_PATH_OVERRIDE=embedded-language-flows/${model}-owt-torch
export ORACLE_TAG_OVERRIDE=elf_archscale_mri_minilm_x8_${model_slug}_${recipe}_ep4_wcmean_seed7_20260906
export RUN_TAG_OVERRIDE=eval_${ORACLE_TAG_OVERRIDE}_exact_val266_seed49
export BERT_DEVICE=cuda
exec bash "$PROJECT_ROOT/submit-jobs/eval_elf_semantic_scaling_x1_exact.sbatch"
