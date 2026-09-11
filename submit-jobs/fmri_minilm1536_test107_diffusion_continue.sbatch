#!/usr/bin/env bash
#SBATCH --job-name=fmri_txt_cont
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-1
#SBATCH --output=submit-jobs/fmri_minilm1536_test107_diffusion_continue-%A_%a.out
#SBATCH --error=submit-jobs/fmri_minilm1536_test107_diffusion_continue-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
parent_run=fmri_minilm1536_to_elf_TEST107_TRANSDUCTIVE_DIFFUSION_OVERFIT_ep300_b16_seed49_20260829
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/$parent_run/best.pt

case "$SLURM_ARRAY_TASK_ID" in
  0)
    export LR=1e-4
    suffix=lr1e4
    ;;
  1)
    export LR=3e-5
    suffix=lr3e5
    ;;
  *) exit 2 ;;
esac

export RUN_TAG=fmri_minilm1536_to_elf_TEST107_TRANSDUCTIVE_DIFFUSION_OVERFIT_continue300_${suffix}_seed49_20260830
export EPOCHS=300
export EVAL_EVERY=100
export RETRIEVAL_EVAL_EVERY=200
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="TRANSDUCTIVE CONTROL continuation, not held-out text generalization. Resume the best fixed-brain 107-row downstream ELF overfit checkpoint at a validation-independent learning rate. MRI2SEM remains fixed and was trained without these four test stories."

exec bash "$PROJECT_ROOT/submit-jobs/fmri_minilm1536_test107_diffusion_overfit.sbatch"
