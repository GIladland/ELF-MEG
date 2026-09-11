#!/usr/bin/env bash
#SBATCH --job-name=fmri_oof_adapt
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=submit-jobs/fmri_minilm_knowntext_oof_adapter-%j.out
#SBATCH --error=submit-jobs/fmri_minilm_knowntext_oof_adapter-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
oracle_run=fmri_minilm_knowntext_exact384_all12098_testfirst_full_k64_ep100_b64_seed49_20260829

export NPZ_PATH=$DATA_ROOT/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz
export SOURCE_NPZ=$NPZ_PATH
export RUN_TAG=${RUN_TAG:-fmri_minilm_knowntext_oofmix1536_adapteronly_ep10_b32_seed49_20260829}
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$known_dir/oofmix_current10_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/$oracle_run/best.pt
export INIT_E2E_ADAPTER_MISMATCH=mean-blocks

export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=266
export EVAL_NUM_EXAMPLES=266
export RETRIEVAL_NUM_EXAMPLES=266
export EVAL_EVERY=1000
export RETRIEVAL_EVAL_EVERY=2000
export RETRIEVAL_BATCH_SIZE=256
export EPOCHS=10
export BATCH_SIZE=32
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_ADAPTER_KIND=flat
export SEMANTIC_INPUT_PROJECTION_DIM=384
export FREEZE_ELF=1
export LAST_N_BLOCKS=-1
export LR=1e-4
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_PROB=0.5
export DECODER_NOISE_SCALE=2.5
export TRAIN_TARGET_MASK_MODE=full
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CHECKPOINT_METRIC=negative_wer
export SEED=49
export USE_WANDB=1
export WANDB_RUN_GROUP=fmri_minilm_knowntext_unseenbrain_20260829
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Stage B1 leakage-safe adaptation. Load the all-sentence exact-MiniLM384 ELF checkpoint, add a mean-block initialized 1536-to-384 projection, freeze ELF, and train only the projection/adapter on equal oracle plus story-level OOF MRI2SEM rows. Select on predicted validation; all 107 test MRI2SEM vectors remain excluded."
export SAVE_FINAL_CHECKPOINT=1

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"

