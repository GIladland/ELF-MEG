#!/usr/bin/env bash
#SBATCH --job-name=fmri_mem_test
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=submit-jobs/eval_fmri_knowntext_freshmem_interface_test107-%j.out
#SBATCH --error=submit-jobs/eval_fmri_knowntext_freshmem_interface_test107-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
checkpoint=${INTERFACE_CHECKPOINT:?Set INTERFACE_CHECKPOINT to the validation-selected interface checkpoint}
winner_tag=${INTERFACE_WINNER_TAG:-winner}
source_dir=$DATA_ROOT/minilm/oracle1536_delayed_seg10_middle10content_clockfix
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1

export NPZ_PATH=$source_dir/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_pred1536_test_middle10content_clockfix.npz
export SOURCE_NPZ=$NPZ_PATH
export RUN_TAG=eval_fmri_knowntext_freshmem_${winner_tag}_unseenbrain_test107_seed49_20260901
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$known_dir/eval_latents/${RUN_TAG}_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=$checkpoint
export INIT_E2E_ADAPTER_MISMATCH=error

export EVAL_ONLY=1
export EPOCHS=0
export NUM_EXAMPLES=107
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=107
export RETRIEVAL_NUM_EXAMPLES=107
export BATCH_SIZE=32
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_ADAPTER_KIND=flat
export SEMANTIC_INPUT_PROJECTION_DIM=384
export FREEZE_ELF=1
export LAST_N_BLOCKS=-1
export ELF_LORA_RANK=0
if [[ "$winner_tag" == *lora8* ]]; then
  export ELF_LORA_RANK=8
  export ELF_LORA_ALPHA=16
  export ELF_LORA_TARGETS=attention,mlp
  export ELF_LORA_LAST_N_BLOCKS=-1
fi
export TRAIN_TARGET_MASK_MODE=full
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CHECKPOINT_METRIC=negative_wer

export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=fmri_knowntext_freshmem_interface_matrix_20260901
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Single post-selection test107 evaluation of leakage-safe interface winner $winner_tag. Test fMRI predictions were absent from oracle diffusion and interface training."
export SEED=49

export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
