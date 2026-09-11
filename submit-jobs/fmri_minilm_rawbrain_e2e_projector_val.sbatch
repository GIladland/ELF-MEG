#!/usr/bin/env bash
#SBATCH --job-name=fmri_e2e_head
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-1
#SBATCH --output=submit-jobs/fmri_minilm_rawbrain_e2e_projector_val-%A_%a.out
#SBATCH --error=submit-jobs/fmri_minilm_rawbrain_e2e_projector_val-%A_%a.err

set -euo pipefail

# True staged E2E development run. Only train_x/val_x are loaded; test_x is closed.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
mri_run=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1
oracle_run=fmri_minilm_knowntext_exact384_all12098_testfirst_full_k64_ep100_b64_seed49_20260829

export SEMANTIC_ADAPTER_KIND=residual_mlp
export SEMANTIC_MAPPER_HIDDEN_DIM=512
export ADAPTER_DROPOUT=0.0
case "$SLURM_ARRAY_TASK_ID" in
  0)
    variant=resmlp_h512
    ;;
  1)
    variant=resdelaytok_h1024_l2_h8
    export SEMANTIC_ADAPTER_KIND=residual_delay_tokens
    export SEMANTIC_MAPPER_HIDDEN_DIM=1024
    export SEMANTIC_DELAY_LAYERS=2
    export SEMANTIC_DELAY_HEADS=8
    export ADAPTER_DROPOUT=0.1
    ;;
  *) exit 2 ;;
esac

export NPZ_PATH=$known_dir/oracle_exact_minilm384_train_val.npz
export SOURCE_NPZ=$NPZ_PATH
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export TRAIN_BALANCE_KEY=story
export BRAIN_INPUT_NPZ=$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz
export BRAIN_TRAIN_KEY=train_x
export BRAIN_VAL_KEY=val_x
export BRAIN_MODEL_CHECKPOINT=$mri_run/model.pt
export BRAIN_SEMANTIC_OUTPUT_DIM=1536
export BRAIN_HIDDEN_DIM=2048
export BRAIN_RES_BLOCKS=4
export BRAIN_DROPOUT=0.1
export BRAIN_UNFREEZE=projector
export BRAIN_LR=1e-5

export SEMANTIC_ALIGNMENT_TARGET_NPZ=$known_dir/oracle_exact_minilm384_train_val.npz
export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings
export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=1.0
export SEMANTIC_ALIGNMENT_LOSS_TYPE=cosine
export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.1
export SEMANTIC_CONTRASTIVE_TEMPERATURE=0.07
export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.1

export RUN_TAG=fmri_minilm_rawbrain_e2e_projector_long100_${variant}_storybal_b32_seed49_20260830
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$known_dir/oracle_exact_minilm384_trainval_current10_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/$oracle_run/best.pt
export INIT_E2E_ADAPTER_MISMATCH=residual-mlp

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
export LR=1e-4
export LR_SCHEDULE=cosine
export WARMUP_STEPS=1000
export MIN_LR_RATIO=0.01
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CHECKPOINT_METRIC=content_word_overlap
export SAVE_FINAL_CHECKPOINT=1
export SAVE_EVAL_CHECKPOINTS=1
export EVAL_CHECKPOINT_TOP_K=1
export EVAL_CHECKPOINT_EVERY=5
export SEED=49
export USE_WANDB=1
export WANDB_RUN_GROUP=fmri_minilm_rawbrain_e2e_projector_val_20260830
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="True raw-fMRI E2E stage 1. Jointly train the MRI2SEM 2048-to-1536 projector at 1e-5 and semantic bridge at 1e-4; MRI2SEM trunk, exact-MiniLM context projector, and ELF-B are frozen. Train11725/val266 only; test107 brain array is never loaded."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
