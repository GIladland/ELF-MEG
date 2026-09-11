#!/usr/bin/env bash
#SBATCH --job-name=fmri_txt_cont3
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/fmri_minilm1536_test107_diffusion_continue3-%j.out
#SBATCH --error=submit-jobs/fmri_minilm1536_test107_diffusion_continue3-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
parent_run=fmri_minilm1536_to_elf_TEST107_TRANSDUCTIVE_DIFFUSION_OVERFIT_continue2_300_lr1e4_seed49_20260830
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/$parent_run/best.pt
export LR=1e-4
export RUN_TAG=fmri_minilm1536_to_elf_TEST107_TRANSDUCTIVE_DIFFUSION_OVERFIT_continue3_300_lr1e4_seed49_20260830
export EPOCHS=300
export EVAL_EVERY=100
export RETRIEVAL_EVAL_EVERY=200
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Third and final TRANSDUCTIVE CONTROL refinement, not held-out text generalization. Resume the WER-0.3963 fixed-brain 107-row downstream ELF checkpoint. MRI2SEM remains frozen and was trained without these four test stories."

exec bash "$PROJECT_ROOT/submit-jobs/fmri_minilm1536_test107_diffusion_overfit.sbatch"
