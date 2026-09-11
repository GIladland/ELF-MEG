#!/usr/bin/env bash
#SBATCH --job-name=sem_zeroshot
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-2
#SBATCH --output=submit-jobs/eval_middle10_semantic_pretrained_zeroshot-%A_%a.out
#SBATCH --error=submit-jobs/eval_middle10_semantic_pretrained_zeroshot-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
text_variant=${TEXT_VARIANT:-middle10_clockfix}
source_dir=${SOURCE_DIR:-$DATA_ROOT/tang_gpt/oracle3072_middle10content_clockfix}
prefix=${SOURCE_PREFIX:-uts01_tang_gpt_lanczos_seg10_oracle3072_middle10content_clockfix}
avg768_dir=${AVG768_DIR:-$DATA_ROOT/tang_gpt/oracle3072_middle10content_clockfix_tangavg768}
minilm_dir=${MINILM_DIR:-$DATA_ROOT/tang_gpt/oracle3072_middle10content_clockfix_minilm}

case "$SLURM_ARRAY_TASK_ID" in
  0)
    variant=tangavg768
    export NPZ_PATH=$avg768_dir/${prefix}_train_val_tangavg768.npz
    export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/tang_gpt_storychunks_decoderprob90_continue_from_current_best_30ep_b32_a100_20260823/best.pt
    ;;
  1)
    variant=minilm384
    export NPZ_PATH=$minilm_dir/${prefix}_train_val_minilm.npz
    export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/tang_gpt_seg10_podcast_storychunks8to18_t12_minilm_elf_fullmask_k64_full_150ep_b32_short12h_a100/best.pt
    ;;
  2)
    variant=native3072_meaninit
    export NPZ_PATH=$source_dir/${prefix}_train_val.npz
    export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/tang_gpt_storychunks_decoderprob90_continue_from_current_best_30ep_b32_a100_20260823/best.pt
    export INIT_E2E_ADAPTER_MISMATCH=mean-blocks
    export SEMANTIC_INPUT_PROJECTION_DIM=768
    ;;
  *)
    echo "Unsupported array task: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

export RUN_TAG=${text_variant}_${variant}_pretrained_zeroshot_val128_20260828
export SOURCE_NPZ=$NPZ_PATH
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$DATA_ROOT/tang_gpt/semantic_bridge_diagnostics/${RUN_TAG}_t5small_latents_fp16.npy
export INIT_E2E_ADAPTER_MISMATCH=${INIT_E2E_ADAPTER_MISMATCH:-error}

export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export TORCH_PIP_SPEC=${TORCH_PIP_SPEC:-torch==2.7.1}
export TORCH_PIP_INDEX_URL=${TORCH_PIP_INDEX_URL:-https://download.pytorch.org/whl/cu118}

export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=266
export EVAL_NUM_EXAMPLES=128
export RETRIEVAL_NUM_EXAMPLES=128
export RETRIEVAL_EVAL_EVERY=1
export RETRIEVAL_BATCH_SIZE=256
export EPOCHS=1
export BATCH_SIZE=16
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_INPUT_PROJECTION_DIM=${SEMANTIC_INPUT_PROJECTION_DIM:-0}
export FREEZE_ELF=1
export EVAL_ONLY=1
export USE_WANDB=0
export SAVE_FINAL_CHECKPOINT=0
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_NOISE_SCALE=2.5

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
