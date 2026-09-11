#!/usr/bin/env bash
#SBATCH --job-name=elf_arch_raw
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-3%2
#SBATCH --output=submit-jobs/eval_elf_architecture_scaling_x1_raw-%A_%a.out
#SBATCH --error=submit-jobs/eval_elf_architecture_scaling_x1_raw-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) modality=mri_minilm; model=ELF-M; model_slug=elfm; export LADDER_TASK_ID_OVERRIDE=4 ;;
  1) modality=mri_minilm; model=ELF-L; model_slug=elfl; export LADDER_TASK_ID_OVERRIDE=4 ;;
  2) modality=meg_ada; model=ELF-M; model_slug=elfm; export LADDER_TASK_ID_OVERRIDE=5 ;;
  3) modality=meg_ada; model=ELF-L; model_slug=elfl; export LADDER_TASK_ID_OVERRIDE=5 ;;
  *) exit 2 ;;
esac

export MODEL_OVERRIDE=$model
export CHECKPOINT_PATH_OVERRIDE=embedded-language-flows/${model}-owt-torch
export ORACLE_TAG_OVERRIDE=elf_archscale_${modality}_x1_${model_slug}_exact_compute44k_content_seed7_20260904
export RUN_TAG_OVERRIDE=eval_elf_archscale_${modality}_x1_${model_slug}_rawbrain_seed49_20260904
export BERT_DEVICE=cuda

exec bash "$PROJECT_ROOT/submit-jobs/eval_elf_semantic_scaling_raw_ladder.sbatch"
