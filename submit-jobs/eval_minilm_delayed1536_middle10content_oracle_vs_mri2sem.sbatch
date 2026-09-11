#!/usr/bin/env bash
#SBATCH --job-name=eval_mini1536_m10
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --array=0-1
#SBATCH --output=submit-jobs/eval_minilm_delayed1536_middle10content_oracle_vs_mri2sem-%A_%a.out
#SBATCH --error=submit-jobs/eval_minilm_delayed1536_middle10content_oracle_vs_mri2sem-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
source_dir=${SOURCE_DIR:-$DATA_ROOT/minilm/oracle1536_delayed_seg10_middle10content_clockfix}
prefix=${SOURCE_PREFIX:-uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_oracle1536_middle10content_clockfix}
bridge_run=${BRIDGE_RUN:-minilm_delayed1536_seg10_middle10content_clockfix_meaninit_freezeelf_adapter_train11725_val266_ep30_b32_20260828}

case "$SLURM_ARRAY_TASK_ID" in
  0)
    condition=oracle_minilm1536
    export NPZ_PATH=$source_dir/${prefix}_test.npz
    ;;
  1)
    condition=mri2sem_pred1536
    export NPZ_PATH=$source_dir/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_pred1536_test_middle10content_clockfix.npz
    ;;
  *)
    echo "Unsupported array task: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

export RUN_TAG=eval_${bridge_run}_${condition}
export SOURCE_NPZ=$NPZ_PATH
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$DATA_ROOT/minilm/eval_latents/${RUN_TAG}_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/$bridge_run/best.pt
export INIT_E2E_ADAPTER_MISMATCH=error

export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export TORCH_PIP_SPEC=${TORCH_PIP_SPEC:-torch==2.7.1}
export TORCH_PIP_INDEX_URL=${TORCH_PIP_INDEX_URL:-https://download.pytorch.org/whl/cu118}

export NUM_EXAMPLES=107
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=107
export RETRIEVAL_NUM_EXAMPLES=107
export RETRIEVAL_EVAL_EVERY=1
export RETRIEVAL_BATCH_SIZE=256
export EPOCHS=1
export BATCH_SIZE=16
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_INPUT_PROJECTION_DIM=${SEMANTIC_INPUT_PROJECTION_DIM:-384}
export SEMANTIC_ADAPTER_KIND=${SEMANTIC_ADAPTER_KIND:-flat}
export SEMANTIC_DELAY_TOKENS=${SEMANTIC_DELAY_TOKENS:-4}
export SEMANTIC_DELAY_LAYERS=${SEMANTIC_DELAY_LAYERS:-2}
export SEMANTIC_DELAY_HEADS=${SEMANTIC_DELAY_HEADS:-8}
export FREEZE_ELF=1
export EVAL_ONLY=1
export USE_WANDB=1
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-minilm_delayed1536_middle10content_paired_test_20260828}
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Paired held-out $condition evaluation of the delayed MiniLM1536 bridge. Oracle and MRI2SEM-predicted inputs share exactly the same 107 rows and audited 10-word targets."
export SAVE_FINAL_CHECKPOINT=0
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_NOISE_SCALE=2.5

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
