#!/usr/bin/env bash
#SBATCH --job-name=known107_elf
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=submit-jobs/train_fmri_minilm_knowntext_target107_exact384-%j.out
#SBATCH --error=submit-jobs/train_fmri_minilm_knowntext_target107_exact384-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
base_run=minilm384_full_unfreeze_continue_werselect_ep75_b16_20260829

export RUN_TAG=${RUN_TAG:-fmri_minilm_knowntext_TARGET107_exact384_full_unfreeze_ep300_b16_seed49_20260829}
export NPZ_PATH=$known_dir/oracle_exact_minilm384_test107.npz
export SOURCE_NPZ=$NPZ_PATH
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$known_dir/${RUN_TAG}_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/$base_run/best.pt
export INIT_E2E_ADAPTER_MISMATCH=error

export NUM_EXAMPLES=107
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=107
export RETRIEVAL_NUM_EXAMPLES=107
export EVAL_EVERY=500
export RETRIEVAL_EVAL_EVERY=500
export RETRIEVAL_BATCH_SIZE=256
export EPOCHS=300
export BATCH_SIZE=16
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_ADAPTER_KIND=flat
export SEMANTIC_INPUT_PROJECTION_DIM=0
export FREEZE_ELF=0
export LAST_N_BLOCKS=-1
export LR=2e-5
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
export WANDB_RUN_GROUP=fmri_minilm_knowntext_unseenbrain_target107_20260829
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Allowed downstream-only overfit: train exact-MiniLM384-to-ELF on the 107 known target texts and exact text embeddings. No MRI2SEM test predictions are used for training, validation, checkpoint selection, or sampling. A fixed story-OOF-trained brain mapper is evaluated only after this job succeeds."
export SAVE_FINAL_CHECKPOINT=1

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
