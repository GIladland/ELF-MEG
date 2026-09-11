#!/usr/bin/env bash
#SBATCH --job-name=elf_meg_ml_raw
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-3%2
#SBATCH --output=submit-jobs/eval_elf_archscale_meg_ml_raw_search_20260906-%A_%a.out
#SBATCH --error=submit-jobs/eval_elf_archscale_meg_ml_raw_search_20260906-%A_%a.err

set -euo pipefail
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) model=ELF-M; model_slug=elfm; recipe=continue_const1e5; epochs=8 ;;
  1) model=ELF-M; model_slug=elfm; recipe=continue_cos2e5; epochs=8 ;;
  2) model=ELF-L; model_slug=elfl; recipe=continue_const5e6; epochs=8 ;;
  3) model=ELF-L; model_slug=elfl; recipe=continue_cos1e5; epochs=8 ;;
  *) exit 2 ;;
esac

export LADDER_TASK_ID_OVERRIDE=5
export MODEL_OVERRIDE=$model
export CHECKPOINT_PATH_OVERRIDE=embedded-language-flows/${model}-owt-torch
export ORACLE_TAG_OVERRIDE=elf_archscale_meg_ada_x16_${model_slug}_${recipe}_ep${epochs}_wcmean_seed7_20260906
export RUN_TAG_OVERRIDE=eval_${ORACLE_TAG_OVERRIDE}_rawbrain_val110_seed49
export BERT_DEVICE=cuda

exec bash "$PROJECT_ROOT/submit-jobs/eval_elf_semantic_scaling_raw_ladder.sbatch"
