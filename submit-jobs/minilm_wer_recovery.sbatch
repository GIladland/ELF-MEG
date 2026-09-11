#!/usr/bin/env bash
#SBATCH --job-name=minilm_wer_recovery
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-1
#SBATCH --output=submit-jobs/minilm_wer_recovery-%A_%a.out
#SBATCH --error=submit-jobs/minilm_wer_recovery-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export TORCH_PIP_SPEC=${TORCH_PIP_SPEC:-torch==2.7.1}
export TORCH_PIP_INDEX_URL=${TORCH_PIP_INDEX_URL:-https://download.pytorch.org/whl/cu118}

delayed_dir=$DATA_ROOT/minilm/oracle1536_delayed_seg10_middle10content_clockfix
delayed_prefix=uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_oracle1536_middle10content_clockfix
delayed_flat_run=minilm_delayed1536_seg10_middle10content_clockfix_meaninit_freezeelf_adapter_train11725_val266_ep30_b32_20260828
exact_dir=$DATA_ROOT/tang_gpt/oracle3072_middle10content_clockfix_minilm
exact_prefix=uts01_tang_gpt_lanczos_seg10_oracle3072_middle10content_clockfix
exact_run=middle10_clockfix_minilm384_full_unfreeze_train11725_val266_ep30_b16_20260828

case "$SLURM_ARRAY_TASK_ID" in
  0)
    export RUN_TAG=minilm_delayed1536_delaytokens4_freezeelf_werselect_ep50_b64_20260829
    export NPZ_PATH=$delayed_dir/${delayed_prefix}_train_val.npz
    export TARGET_LATENTS_CACHE=$delayed_dir/${delayed_flat_run}_t5small_latents_fp16.npy
    export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/$delayed_flat_run/best.pt
    export INIT_E2E_ADAPTER_MISMATCH=skip
    export SEMANTIC_ADAPTER_KIND=delay_tokens
    export SEMANTIC_INPUT_PROJECTION_DIM=0
    export SEMANTIC_DELAY_TOKENS=4
    export SEMANTIC_DELAY_LAYERS=2
    export SEMANTIC_DELAY_HEADS=8
    export FREEZE_ELF=1
    export LAST_N_BLOCKS=-1
    export EPOCHS=50
    export BATCH_SIZE=64
    export LR=1e-4
    export WANDB_NOTES="WER-focused delayed MiniLM1536 oracle bridge. Preserve delay1-4 as four ordered semantic tokens, freeze ELF, and select best.pt by deterministic predicted-validation WER."
    ;;
  1)
    export RUN_TAG=minilm384_full_unfreeze_continue_werselect_ep75_b16_20260829
    export NPZ_PATH=$exact_dir/${exact_prefix}_train_val_minilm.npz
    export TARGET_LATENTS_CACHE=$DATA_ROOT/tang_gpt/semantic_bridge_full/${exact_run}_t5small_latents_fp16.npy
    export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/$exact_run/best.pt
    export INIT_E2E_ADAPTER_MISMATCH=error
    export SEMANTIC_ADAPTER_KIND=flat
    export SEMANTIC_INPUT_PROJECTION_DIM=0
    export FREEZE_ELF=0
    export LAST_N_BLOCKS=-1
    export EPOCHS=75
    export BATCH_SIZE=16
    export LR=2e-5
    export WANDB_NOTES="WER upper-bound continuation from the exact middle-10-word MiniLM384 embedding. Fully unfreeze ELF for 75 more epochs and select best.pt by deterministic held-out validation WER."
    ;;
  *)
    echo "Unsupported array task: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

export SOURCE_NPZ=$NPZ_PATH
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=266
export EVAL_NUM_EXAMPLES=266
export RETRIEVAL_NUM_EXAMPLES=266
export RETRIEVAL_EVAL_EVERY=2000
export RETRIEVAL_BATCH_SIZE=256
export EVAL_EVERY=500
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_PROB=0.5
export DECODER_NOISE_SCALE=2.5
export TRAIN_TARGET_MASK_MODE=full
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CHECKPOINT_METRIC=negative_wer
export USE_WANDB=1
export WANDB_RUN_GROUP=minilm_to_elf_wer_recovery_20260829
export WANDB_RUN_NAME=$RUN_TAG
export SAVE_FINAL_CHECKPOINT=1

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
