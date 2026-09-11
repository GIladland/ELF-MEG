#!/usr/bin/env bash
#SBATCH --job-name=eval_valcal_mri
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --array=0-1
#SBATCH --output=submit-jobs/eval_valcal_mri2sem_minilm384_pretrained_elf-%A_%a.out
#SBATCH --error=submit-jobs/eval_valcal_mri2sem_minilm384_pretrained_elf-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
case "$SLURM_ARRAY_TASK_ID" in
  0) mapper_variant=mlp ;;
  1) mapper_variant=linear ;;
  *) exit 2 ;;
esac

mapper_dir=/data/engs-pnpl/glandau/elf-runs/tang3072_to_minilm384_center10_clockfix_${mapper_variant}_valpred_calibrated_20260828
export NPZ_PATH=$mapper_dir/mapped_mri2sem_test_minilm384_valcal.npz
export RUN_TAG=eval_center10_clockfix_mri2sem_pred3072_${mapper_variant}_valcal_minilm384_pretrained_elf_20260828
export SOURCE_NPZ=$NPZ_PATH
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$DATA_ROOT/tang_gpt/semantic_bridge_diagnostics/${RUN_TAG}_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/tang_gpt_seg10_podcast_storychunks8to18_t12_minilm_elf_fullmask_k64_full_150ep_b32_short12h_a100/best.pt
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
export WANDB_RUN_GROUP=center10_clockfix_valcal_mri2sem_to_minilm_to_elf_20260828
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Held-out 107-row MRI2SEM evaluation after $mapper_variant Tang-to-MiniLM mapping and a correction fit only on the separate 266-row MRI2SEM validation predictions. Test labels were not used to choose the correction."
export SAVE_FINAL_CHECKPOINT=0
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_NOISE_SCALE=2.5

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
