#!/usr/bin/env bash
#SBATCH --job-name=sem_full_bridge
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --array=0-2
#SBATCH --output=submit-jobs/middle10_semantic_full_bridges-%A_%a.out
#SBATCH --error=submit-jobs/middle10_semantic_full_bridges-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
text_variant=${TEXT_VARIANT:-middle10_clockfix}
source_dir=${SOURCE_DIR:-$DATA_ROOT/tang_gpt/oracle3072_middle10content_clockfix}
prefix=${SOURCE_PREFIX:-uts01_tang_gpt_lanczos_seg10_oracle3072_middle10content_clockfix}
avg768_dir=${AVG768_DIR:-$DATA_ROOT/tang_gpt/oracle3072_middle10content_clockfix_tangavg768}
minilm_dir=${MINILM_DIR:-$DATA_ROOT/tang_gpt/oracle3072_middle10content_clockfix_minilm}
tang_checkpoint=/data/engs-pnpl/glandau/elf-runs/tang_gpt_storychunks_decoderprob90_continue_from_current_best_30ep_b32_a100_20260823/best.pt
minilm_checkpoint=/data/engs-pnpl/glandau/elf-runs/tang_gpt_seg10_podcast_storychunks8to18_t12_minilm_elf_fullmask_k64_full_150ep_b32_short12h_a100/best.pt

case "$SLURM_ARRAY_TASK_ID" in
  0)
    variant=native3072_meaninit
    export NPZ_PATH=$source_dir/${prefix}_train_val.npz
    export INIT_E2E_CHECKPOINT=$tang_checkpoint
    export INIT_E2E_ADAPTER_MISMATCH=mean-blocks
    export SEMANTIC_INPUT_PROJECTION_DIM=768
    ;;
  1)
    variant=tangavg768
    export NPZ_PATH=$avg768_dir/${prefix}_train_val_tangavg768.npz
    export INIT_E2E_CHECKPOINT=$tang_checkpoint
    export INIT_E2E_ADAPTER_MISMATCH=error
    ;;
  2)
    variant=minilm384
    export NPZ_PATH=$minilm_dir/${prefix}_train_val_minilm.npz
    export INIT_E2E_CHECKPOINT=$minilm_checkpoint
    export INIT_E2E_ADAPTER_MISMATCH=error
    ;;
  *)
    echo "Unsupported array task: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

export RUN_TAG=${text_variant}_${variant}_full_unfreeze_train11725_val266_ep30_b16_20260828
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export SOURCE_NPZ=$NPZ_PATH
export TARGET_LATENTS_CACHE=$DATA_ROOT/tang_gpt/semantic_bridge_full/${RUN_TAG}_t5small_latents_fp16.npy

export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export TORCH_PIP_SPEC=${TORCH_PIP_SPEC:-torch==2.7.1}
export TORCH_PIP_INDEX_URL=${TORCH_PIP_INDEX_URL:-https://download.pytorch.org/whl/cu118}

export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=266
export EVAL_NUM_EXAMPLES=266
export RETRIEVAL_NUM_EXAMPLES=266
export RETRIEVAL_EVAL_EVERY=4000
export RETRIEVAL_BATCH_SIZE=256
export EVAL_EVERY=1000
export EPOCHS=30
export BATCH_SIZE=16
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_INPUT_PROJECTION_DIM=${SEMANTIC_INPUT_PROJECTION_DIM:-0}
export FREEZE_ELF=0
export LAST_N_BLOCKS=-1
export LR=2e-5
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_NOISE_SCALE=2.5
export TRAIN_TARGET_MASK_MODE=full
export USE_WANDB=1
export WANDB_RUN_GROUP=${text_variant}_semantic_full_bridges_20260828
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Held-out semantic-to-ELF bridge for $variant on 11,725 audited 10-word rows with a 266-row validation tail. Full ELF fine-tuning at low LR after the corresponding semantic-family initialization."
export SAVE_FINAL_CHECKPOINT=1

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
