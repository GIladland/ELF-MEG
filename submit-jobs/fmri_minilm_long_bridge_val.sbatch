#!/usr/bin/env bash
#SBATCH --job-name=fmri_long_bridge
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-4
#SBATCH --output=submit-jobs/fmri_minilm_long_bridge_val-%A_%a.out
#SBATCH --error=submit-jobs/fmri_minilm_long_bridge_val-%A_%a.err

set -euo pipefail

# Development-only long sweep. The final test107 brain examples are never loaded.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
oracle_run=fmri_minilm_knowntext_exact384_all12098_testfirst_full_k64_ep100_b64_seed49_20260829

export SEMANTIC_ADAPTER_KIND=residual_mlp
export INIT_E2E_ADAPTER_MISMATCH=residual-mlp
export SEMANTIC_MAPPER_HIDDEN_DIM=512
export ADAPTER_DROPOUT=0.0
export LR=1e-4
export LR_SCHEDULE=constant
export WARMUP_STEPS=0
export MIN_LR_RATIO=0.01

case "$SLURM_ARRAY_TASK_ID" in
  0)
    variant=resmlp_h512_constant
    ;;
  1)
    variant=resmlp_h512_cosine
    export LR_SCHEDULE=cosine
    export WARMUP_STEPS=1000
    ;;
  2)
    variant=resmlp_h1024_drop0p1_cosine
    export SEMANTIC_MAPPER_HIDDEN_DIM=1024
    export ADAPTER_DROPOUT=0.1
    export LR_SCHEDULE=cosine
    export WARMUP_STEPS=1000
    ;;
  3)
    variant=resdelaytok_h1024_l2_h8_drop0p1_cosine
    export SEMANTIC_ADAPTER_KIND=residual_delay_tokens
    export SEMANTIC_MAPPER_HIDDEN_DIM=1024
    export SEMANTIC_DELAY_LAYERS=2
    export SEMANTIC_DELAY_HEADS=8
    export ADAPTER_DROPOUT=0.1
    export LR_SCHEDULE=cosine
    export WARMUP_STEPS=1000
    ;;
  4)
    variant=resmlp_h512_curriculum20_cosine
    export LR_SCHEDULE=cosine
    export WARMUP_STEPS=1000
    export CURRICULUM_SEMANTIC_ONLY_EPOCHS=20
    export CURRICULUM_DENOISER_LOSS_WEIGHT=0.1
    export CURRICULUM_DECODER_LOSS_WEIGHT=0.1
    ;;
  *) exit 2 ;;
esac

export NPZ_PATH=$DATA_ROOT/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz
export SOURCE_NPZ=$NPZ_PATH
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export TRAIN_CONDITION_SOURCES=oof_prediction
export CONDITION_SOURCE_KEY=condition_source
export TRAIN_BALANCE_KEY=story
export SEMANTIC_ALIGNMENT_TARGET_NPZ=$known_dir/oofmix_exact_minilm384_targets_train23450_val266.npz
export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings
export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=1.0
export SEMANTIC_ALIGNMENT_LOSS_TYPE=cosine
export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.1
export SEMANTIC_CONTRASTIVE_TEMPERATURE=0.07
export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.1

export RUN_TAG=fmri_minilm_long100_${variant}_oofonly_storybal_b32_seed49_20260830
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$known_dir/oofmix_current10_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/$oracle_run/best.pt

export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export SKIP_ARC_SETUP=1
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=266
export EVAL_NUM_EXAMPLES=266
export RETRIEVAL_NUM_EXAMPLES=266
export EVAL_EVERY=1000
export RETRIEVAL_EVAL_EVERY=1000
export RETRIEVAL_BATCH_SIZE=256
export EPOCHS=100
export BATCH_SIZE=32
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_DELAY_TOKENS=4
export SEMANTIC_INPUT_PROJECTION_DIM=0
export NORMALIZE_SEMANTIC_PROJECTION=0
export TRAIN_SEMANTIC_INPUT_ONLY=1
export FREEZE_ELF=1
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CHECKPOINT_METRIC=content_word_overlap
export SAVE_FINAL_CHECKPOINT=1
export SAVE_EVAL_CHECKPOINTS=1
export EVAL_CHECKPOINT_TOP_K=1
export EVAL_CHECKPOINT_EVERY=5
export SEED=49
export USE_WANDB=1
export WANDB_RUN_GROUP=fmri_minilm_long_bridge_val_20260830
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Long validation-only fMRI bridge sweep. Train on 11,725 story-balanced OOF brain predictions and select only on pred-val266. Test107 brain vectors remain closed. Retain Pareto checkpoints for overlap, retrieval, structure, and WER."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
