#!/usr/bin/env bash
#SBATCH --job-name=elf_mri_ml_x8adapt
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-7%4
#SBATCH --output=submit-jobs/train_elf_archscale_mri_ml_x8_adapter_search_20260906-%A_%a.out
#SBATCH --error=submit-jobs/train_elf_archscale_mri_ml_x8_adapter_search_20260906-%A_%a.err

set -euo pipefail
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}

task=${SLURM_ARRAY_TASK_ID:-0}
oracle_index=$((task / 2))
adapter_index=$((task % 2))

case "$oracle_index" in
  0) model=ELF-M; model_slug=elfm; recipe=x8_const2e5 ;;
  1) model=ELF-M; model_slug=elfm; recipe=x8_cos1e5 ;;
  2) model=ELF-L; model_slug=elfl; recipe=x8_const1e5 ;;
  3) model=ELF-L; model_slug=elfl; recipe=x8_cos5e6 ;;
  *) exit 2 ;;
esac

case "$adapter_index" in
  0) adapter=residual_mlp; mismatch=residual-mlp; adapter_label=resmlp ;;
  1) adapter=residual_delay_tokens; mismatch=residual-mlp; adapter_label=delaytokens ;;
  *) exit 2 ;;
esac

oracle_tag=elf_archscale_mri_minilm_x8_${model_slug}_${recipe}_ep4_wcmean_seed7_20260906
export LADDER_TASK_ID_OVERRIDE=4
export MODEL_OVERRIDE=$model
export CHECKPOINT_PATH_OVERRIDE=embedded-language-flows/${model}-owt-torch
export ORACLE_TAG_OVERRIDE=$oracle_tag
export RUN_TAG_OVERRIDE=${oracle_tag}_${adapter_label}_adapter_val266_wcmean_seed49
export RETRIEVAL_NUM_EXAMPLES_OVERRIDE=266
export BATCH_SIZE_OVERRIDE=16
export EPOCHS_OVERRIDE=100
export EVAL_EVERY_OVERRIDE=1000
export RETRIEVAL_EVAL_EVERY_OVERRIDE=1000
export SAVE_FINAL_CHECKPOINT_OVERRIDE=0
export SAVE_EVAL_CHECKPOINTS_OVERRIDE=0
export CHECKPOINT_METRIC_OVERRIDE=word_content_mean
export SEMANTIC_ADAPTER_KIND_OVERRIDE=$adapter
export INIT_E2E_ADAPTER_MISMATCH_OVERRIDE=$mismatch
export TRAIN_SEMANTIC_INPUT_ONLY_OVERRIDE=1

exec bash "$PROJECT_ROOT/submit-jobs/train_elf_semantic_scaling_adapter_ladder.sbatch"
