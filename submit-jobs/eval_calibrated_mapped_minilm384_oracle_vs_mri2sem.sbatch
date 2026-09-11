#!/usr/bin/env bash
#SBATCH --job-name=eval_cal_map
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --array=0-1
#SBATCH --output=submit-jobs/eval_calibrated_mapped_minilm384_oracle_vs_mri2sem-%A_%a.out
#SBATCH --error=submit-jobs/eval_calibrated_mapped_minilm384_oracle_vs_mri2sem-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
mapper_variant=${MAPPER_VARIANT:-mlp}
mapper_dir=/data/engs-pnpl/glandau/elf-runs/tang3072_to_minilm384_center10_clockfix_${mapper_variant}_20260828
bridge_run=center10_clockfix_tangmapped_${mapper_variant}_minilm384_freezeelf_adapter_train11725_val266_ep10_20260828

case "$SLURM_ARRAY_TASK_ID" in
  0)
    condition=oracle_tang3072_mapped_minilm384
    export NPZ_PATH=$mapper_dir/mapped_oracle_test_minilm384.npz
    ;;
  1)
    condition=mri2sem_pred3072_mapped_minilm384
    export NPZ_PATH=$mapper_dir/mapped_mri2sem_test_minilm384.npz
    ;;
  *)
    echo "Unsupported array task: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

export RUN_TAG=eval_${bridge_run}_${condition}
export SOURCE_NPZ=$NPZ_PATH
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$DATA_ROOT/tang_gpt/semantic_bridge_diagnostics/${RUN_TAG}_t5small_latents_fp16.npy
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
export SEMANTIC_INPUT_PROJECTION_DIM=0
export FREEZE_ELF=1
export EVAL_ONLY=1
export USE_WANDB=1
export WANDB_RUN_GROUP=center10_clockfix_calibrated_tangmapped_minilm384_paired_test_20260828
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Paired held-out $condition test after full-train calibration of the pretrained MiniLM adapter on $mapper_variant Tang-3072→MiniLM outputs. ELF itself remains frozen."
export SAVE_FINAL_CHECKPOINT=0
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_NOISE_SCALE=2.5

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
