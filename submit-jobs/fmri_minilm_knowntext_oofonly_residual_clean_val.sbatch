#!/usr/bin/env bash
#SBATCH --job-name=fmri_oofclean
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-2
#SBATCH --output=submit-jobs/fmri_minilm_knowntext_oofonly_residual_clean_val-%A_%a.out
#SBATCH --error=submit-jobs/fmri_minilm_knowntext_oofonly_residual_clean_val-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
oracle_run=fmri_minilm_knowntext_exact384_all12098_testfirst_full_k64_ep100_b64_seed49_20260829

export SEMANTIC_ADAPTER_KIND=residual_mlp
export SEMANTIC_INPUT_PROJECTION_DIM=0
export INIT_E2E_ADAPTER_MISMATCH=residual-mlp
export SEMANTIC_MAPPER_HIDDEN_DIM=512
export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=1.0
export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.1
export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.1
export NORMALIZE_SEMANTIC_PROJECTION=0

case "$SLURM_ARRAY_TASK_ID" in
  0)
    variant=h512_norm_lr1e4
    export NORMALIZE_SEMANTIC_PROJECTION=1
    export LR=1e-4
    ;;
  1)
    variant=h512_norm_lr5e4
    export NORMALIZE_SEMANTIC_PROJECTION=1
    export LR=5e-4
    ;;
  2)
    variant=h512_lr1e4
    export LR=1e-4
    ;;
  *) exit 2 ;;
esac

export NPZ_PATH=$DATA_ROOT/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz
export SOURCE_NPZ=$NPZ_PATH
export TRAIN_CONDITION_SOURCES=oof_prediction
export CONDITION_SOURCE_KEY=condition_source
export SEMANTIC_ALIGNMENT_TARGET_NPZ=$known_dir/oofmix_exact_minilm384_targets_train23450_val266.npz
export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings
export SEMANTIC_ALIGNMENT_LOSS_TYPE=cosine
export SEMANTIC_CONTRASTIVE_TEMPERATURE=0.07
export RUN_TAG=fmri_minilm_knowntext_oofonly_CLEAN_${variant}_inputonly_ep10_b32_seed49_20260830
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$known_dir/oofmix_current10_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/$oracle_run/best.pt

export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=266
export EVAL_NUM_EXAMPLES=266
export RETRIEVAL_NUM_EXAMPLES=266
export EVAL_EVERY=1000
export RETRIEVAL_EVAL_EVERY=2000
export RETRIEVAL_BATCH_SIZE=256
export EPOCHS=10
export BATCH_SIZE=32
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_DELAY_TOKENS=4
export ADAPTER_DROPOUT=0.0
export TRAIN_SEMANTIC_INPUT_ONLY=1
export FREEZE_ELF=1
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CHECKPOINT_METRIC=negative_wer
export SEED=49
export USE_WANDB=1
export WANDB_RUN_GROUP=fmri_minilm_knowntext_oofonly_CLEAN_residual_val_20260830
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Clean rerun after rejecting raced artifacts. Unique output directories; train only on 11,725 story-OOF MRI2SEM predictions, validate on separate pred-val266, freeze ELF, exclude test107. No dependent test job."
export SAVE_FINAL_CHECKPOINT=1

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
