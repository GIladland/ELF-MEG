#!/usr/bin/env bash
#SBATCH --job-name=qc4_elfm_flxa
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=2-3%1
#SBATCH --output=submit-jobs/train_qc4wyals_elfm_semantic_xattn_flowlatent_20260912-%A_%a.out
#SBATCH --error=submit-jobs/train_qc4wyals_elfm_semantic_xattn_flowlatent_20260912-%A_%a.err

set -euo pipefail

# Train decoder-only semantic cross-attention on latents generated from noise by
# the frozen deployment flow, rather than on corruptions of target-text latents.
# The cache contains the 2,652 training rows only. Validation still starts from
# fresh noise, and protected test26 is absent.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
export SOURCE_NPZ_OVERRIDE=$DATA_ROOT/tang_themoth_qc4wyals_nested_oof_20260908/interface/qc4wyals_oof_predicted_train_valtail_ada002.npz
export SOURCE_LABEL=nestedoof_flowdirect
export DECODER_FLOW_LATENT_CACHE=$DATA_ROOT/tang_themoth_qc4wyals_nested_oof_20260908/interface/elfm_frozenflow_train2652_seed66_steps32_target_latents_fp16.npy
export DECODER_FLOW_LATENT_BATCH_SIZE=16
export DECODER_FLOW_LATENT_SEED=66
export DECODER_FLOW_LATENT_DIRECT=1
export EPOCHS_OVERRIDE=${EPOCHS_OVERRIDE:-10}

exec bash "$PROJECT_ROOT/submit-jobs/train_qc4wyals_elfm_semantic_xattn_20260912.sbatch"
