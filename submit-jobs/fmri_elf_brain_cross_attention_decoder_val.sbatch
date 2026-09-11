#!/usr/bin/env bash
#SBATCH --job-name=fmri_xattn_dec
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-3%2
#SBATCH --output=submit-jobs/fmri_elf_brain_cross_attention_decoder_val-%A_%a.out
#SBATCH --error=submit-jobs/fmri_elf_brain_cross_attention_decoder_val-%A_%a.err

set -euo pipefail

# Closest ELF analogue of the frozen T5-prefix baseline: retain the recovered
# diffusion trajectory exactly and allow target tokens to cross-attend to the
# matched brain condition only in ELF's discrete decoder pass.  The inherited
# task arms screen 2/4 blocks, pairing loss, and 1e-4/3e-5 learning rates.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export ELF_BRAIN_CROSS_ATTENTION_DECODER_ONLY=1
export DENOISER_LOSS_WEIGHT_OVERRIDE=0.0
export EPOCHS_OVERRIDE=${EPOCHS_OVERRIDE:-8}
export CROSS_RUN_SUFFIX=${CROSS_RUN_SUFFIX_OVERRIDE:-_decoderonly_8ep}
export WANDB_RUN_GROUP=fmri_elf_brain_cross_attention_decoder_val_20260831
export WANDB_NOTES="Leakage-safe raw-fMRI ELF decoder-only cross-attention screen. Recovered diffusion path, MRI2SEM, semantic adapter, ELF base, and imported LoRA remain frozen. Only zero-init target-to-brain cross-attention trains through decoder CE/pairing on train11725; selection on val266; test107 never loaded. Promotion target content F1 > frozen T5-prefix 0.043743, followed by matched-versus-deranged controls."

exec bash "$PROJECT_ROOT/submit-jobs/fmri_elf_brain_cross_attention_val.sbatch"
