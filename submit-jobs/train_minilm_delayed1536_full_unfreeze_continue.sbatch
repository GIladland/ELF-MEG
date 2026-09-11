#!/usr/bin/env bash
#SBATCH --job-name=minilm1536_unfrz
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=submit-jobs/train_minilm_delayed1536_full_unfreeze_continue-%j.out
#SBATCH --error=submit-jobs/train_minilm_delayed1536_full_unfreeze_continue-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export TORCH_PIP_SPEC=${TORCH_PIP_SPEC:-torch==2.7.1}
export TORCH_PIP_INDEX_URL=${TORCH_PIP_INDEX_URL:-https://download.pytorch.org/whl/cu118}

delayed_dir=$DATA_ROOT/minilm/oracle1536_delayed_seg10_middle10content_clockfix
delayed_prefix=uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_oracle1536_middle10content_clockfix
frozen_run=minilm_delayed1536_delaytokens4_freezeelf_werselect_ep50_b64_20260829

export RUN_TAG=${RUN_TAG:-minilm_delayed1536_delaytokens4_full_unfreeze_continue_werselect_ep12_b16_20260829}
export NPZ_PATH=$delayed_dir/${delayed_prefix}_train_val.npz
export SOURCE_NPZ=$NPZ_PATH
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$delayed_dir/minilm_delayed1536_seg10_middle10content_clockfix_meaninit_freezeelf_adapter_train11725_val266_ep30_b32_20260828_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/$frozen_run/best.pt
export INIT_E2E_ADAPTER_MISMATCH=error

export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=266
export EVAL_NUM_EXAMPLES=266
export RETRIEVAL_NUM_EXAMPLES=266
export RETRIEVAL_EVAL_EVERY=2000
export RETRIEVAL_BATCH_SIZE=256
export EVAL_EVERY=1000
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_ADAPTER_KIND=delay_tokens
export SEMANTIC_INPUT_PROJECTION_DIM=0
export SEMANTIC_DELAY_TOKENS=4
export SEMANTIC_DELAY_LAYERS=2
export SEMANTIC_DELAY_HEADS=8
export FREEZE_ELF=0
export LAST_N_BLOCKS=-1
export EPOCHS=${EPOCHS:-12}
export BATCH_SIZE=${BATCH_SIZE:-16}
export LR=${LR:-2e-5}
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_PROB=0.5
export DECODER_NOISE_SCALE=2.5
export TRAIN_TARGET_MASK_MODE=full
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CHECKPOINT_METRIC=${CHECKPOINT_METRIC:-negative_wer}
export SAVE_FINAL_CHECKPOINT=1
export USE_WANDB=1
export WANDB_RUN_GROUP=minilm_delayed1536_oracle_full_unfreeze_20260829
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Leakage-free stage 2 for delayed MiniLM1536: initialize the four-delay-token adapter from its oracle-train/validation-selected frozen-ELF checkpoint, fully unfreeze ELF at low LR for 12 epochs, train on 11,725 oracle rows, and select only on the 266-row oracle validation split. The 107 brain-test rows are not used."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
