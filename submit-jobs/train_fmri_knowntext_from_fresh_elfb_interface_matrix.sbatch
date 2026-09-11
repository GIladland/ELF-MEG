#!/usr/bin/env bash
#SBATCH --job-name=fmri_mem_if
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-4
#SBATCH --output=submit-jobs/train_fmri_knowntext_from_fresh_elfb_interface_matrix-%A_%a.out
#SBATCH --error=submit-jobs/train_fmri_knowntext_from_fresh_elfb_interface_matrix-%A_%a.err

set -euo pipefail

# Leakage-safe downstream adaptation of an exact-MiniLM/text-memorized ELF.
# Train rows contain exact oracle plus story-OOF MRI2SEM predictions; the final
# 266 rows are the fixed predicted validation split.  Test107 is absent.

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
snapshot=${ORACLE_SNAPSHOT:?Set ORACLE_SNAPSHOT to the immutable exact-MiniLM ELF checkpoint}

export NPZ_PATH=$DATA_ROOT/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz
export SOURCE_NPZ=$NPZ_PATH
export TARGET_LATENTS_CACHE=$known_dir/oofmix_current10_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=$snapshot
export INIT_E2E_ADAPTER_MISMATCH=mean-blocks
export SEMANTIC_INPUT_PROJECTION_DIM=384

export FREEZE_ELF=1
export LAST_N_BLOCKS=-1
export ELF_LORA_RANK=0
export ELF_LORA_ALPHA=16
export ELF_LORA_DROPOUT=0.05
export ELF_LORA_TARGETS=attention,mlp
export ELF_LORA_LAST_N_BLOCKS=-1
export ELF_LR=0
export LR=1e-4
export EPOCHS=20
export BATCH_SIZE=32

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    variant=frozenelf_adapter_lr1e4
    ;;
  1)
    variant=frozenelf_adapter_lr1e5
    export LR=1e-5
    ;;
  2)
    variant=frozenelf_lora8_attnmlp_adapter_lr1e5
    export ELF_LORA_RANK=8
    export ELF_LORA_ALPHA=16
    export ELF_LR=1e-5
    export LR=1e-5
    ;;
  3)
    variant=last4_adapter_lr5e6
    export FREEZE_ELF=0
    export LAST_N_BLOCKS=4
    export ELF_LR=5e-6
    export LR=5e-6
    export BATCH_SIZE=16
    ;;
  4)
    variant=full_elf_adapter_lr1e6
    export FREEZE_ELF=0
    export LAST_N_BLOCKS=-1
    export ELF_LR=1e-6
    export LR=1e-6
    export BATCH_SIZE=16
    ;;
  *) exit 2 ;;
esac

snapshot_tag=${ORACLE_SNAPSHOT_TAG:-interim}
export RUN_TAG=fmri_knowntext_freshmem_${snapshot_tag}_${variant}_val266_seed49_20260901
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG

export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=266
export EVAL_NUM_EXAMPLES=266
export RETRIEVAL_NUM_EXAMPLES=266
export EVAL_EVERY=1000
export RETRIEVAL_EVAL_EVERY=1000
export RETRIEVAL_BATCH_SIZE=256
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_ADAPTER_KIND=flat
export ADAPTER_DROPOUT=0.0
export TRAIN_TARGET_MASK_MODE=full
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CFG_SCALE=1.0
export SELF_COND_CFG_SCALE=1.0
export COND_DROPOUT_PROB=0.0
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_CONTENT_TOKEN_WEIGHT=1.0
export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.0
export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.0
export DECODER_NOISE_SCALE=2.5
export WEIGHT_DECAY=0.0
export LR_SCHEDULE=cosine
export WARMUP_STEPS=500
export MIN_LR_RATIO=0.01
export CHECKPOINT_METRIC=negative_wer
export SAVE_FINAL_CHECKPOINT=1
export SAVE_EVAL_CHECKPOINTS=1
export EVAL_CHECKPOINT_TOP_K=3
export EVAL_CHECKPOINT_EVERY=1

export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=fmri_knowntext_freshmem_interface_matrix_20260901
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_SAMPLE_EXAMPLES=16
export WANDB_NOTES="Leakage-safe interface optimization from exact-text-memorized ELF snapshot $snapshot_tag. Variant=$variant. Training uses exact oracle plus story-OOF MRI2SEM train rows and selects on predicted val266. No test107 fMRI vector is loaded."
export SEED=49

export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
