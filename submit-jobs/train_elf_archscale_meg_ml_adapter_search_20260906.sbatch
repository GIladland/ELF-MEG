#!/usr/bin/env bash
#SBATCH --job-name=elf_meg_ml_adapt
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-7%4
#SBATCH --output=submit-jobs/train_elf_archscale_meg_ml_adapter_search_20260906-%A_%a.out
#SBATCH --error=submit-jobs/train_elf_archscale_meg_ml_adapter_search_20260906-%A_%a.err

set -euo pipefail

# For every recovered ADA oracle compare the established flat context adapter
# with a semantic-space residual adapter. ELF stays frozen; predicted MEG val110
# is selection-only and protected test26 is absent.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}

task=${SLURM_ARRAY_TASK_ID:-0}
oracle_index=$((task / 2))
adapter_index=$((task % 2))

case "$oracle_index" in
  0) model=ELF-M; model_slug=elfm; recipe=continue_const1e5; epochs=8 ;;
  1) model=ELF-M; model_slug=elfm; recipe=continue_cos2e5; epochs=8 ;;
  2) model=ELF-L; model_slug=elfl; recipe=continue_const5e6; epochs=8 ;;
  3) model=ELF-L; model_slug=elfl; recipe=continue_cos1e5; epochs=8 ;;
  *) exit 2 ;;
esac

case "$adapter_index" in
  0)
    adapter=flat; mismatch=error; adapter_label=flat
    train_input_only=0; mapper_hidden=512
    ;;
  1)
    adapter=residual_identity; mismatch=direct-identity; adapter_label=resid
    train_input_only=1; mapper_hidden=1024
    ;;
  *) exit 2 ;;
esac

oracle_tag=elf_archscale_meg_ada_x16_${model_slug}_${recipe}_ep${epochs}_wcmean_seed7_20260906
export LADDER_TASK_ID_OVERRIDE=5
export MODEL_OVERRIDE=$model
export CHECKPOINT_PATH_OVERRIDE=embedded-language-flows/${model}-owt-torch
export ORACLE_TAG_OVERRIDE=$oracle_tag
export RUN_TAG_OVERRIDE=${oracle_tag}_${adapter_label}_adapter_val110_wcmean_seed49
export RETRIEVAL_NUM_EXAMPLES_OVERRIDE=110
export BATCH_SIZE_OVERRIDE=32
export EPOCHS_OVERRIDE=100
export EVAL_EVERY_OVERRIDE=250
export RETRIEVAL_EVAL_EVERY_OVERRIDE=250
export SAVE_FINAL_CHECKPOINT_OVERRIDE=0
export SAVE_EVAL_CHECKPOINTS_OVERRIDE=0
export CHECKPOINT_METRIC_OVERRIDE=word_content_mean
export SEMANTIC_ADAPTER_KIND_OVERRIDE=$adapter
export SEMANTIC_MAPPER_HIDDEN_DIM_OVERRIDE=$mapper_hidden
export INIT_E2E_ADAPTER_MISMATCH_OVERRIDE=$mismatch
export TRAIN_SEMANTIC_INPUT_ONLY_OVERRIDE=$train_input_only

exec bash "$PROJECT_ROOT/submit-jobs/train_elf_semantic_scaling_adapter_ladder.sbatch"
