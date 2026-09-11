#!/usr/bin/env bash
#SBATCH --job-name=fmri_s2_fullchain
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-7
#SBATCH --output=submit-jobs/fmri_minilm_stage2_fullchain_e2e_val-%A_%a.out
#SBATCH --error=submit-jobs/fmri_minilm_stage2_fullchain_e2e_val-%A_%a.err

set -euo pipefail

# Genuine full-chain development: train MRI projector, ordered-delay mapper,
# MiniLM-to-ELF context projector, and optionally ELF LoRA. test107 is closed.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
mri_run=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1
baseline_run=fmri_minilm_rawbrain_e2e_projector_long100_resdelaytok_h1024_l2_h8_storybal_b32_seed49_20260830
baseline_checkpoint=/data/engs-pnpl/glandau/elf-runs/$baseline_run/best.pt

export FREEZE_ELF=1
export LAST_N_BLOCKS=-1
export ELF_LORA_RANK=0
export ELF_LORA_ALPHA=16
export ELF_LORA_DROPOUT=0.05
export ELF_LORA_TARGETS=attention
export ELF_LORA_LAST_N_BLOCKS=-1
export ELF_LR=0
export BATCH_SIZE=32
export DECODER_CONTENT_TOKEN_WEIGHT=1.0
export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.0
export DECODER_LOSS_WEIGHT=1.0
export COND_DROPOUT_PROB=0.05

case "$SLURM_ARRAY_TASK_ID" in
  0)
    variant=fulladapter_frozenelf
    ;;
  1)
    variant=fulladapter_lora_attnmlp_r8
    export ELF_LORA_RANK=8
    export ELF_LORA_ALPHA=16
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LR=1e-5
    ;;
  2)
    variant=fulladapter_lora_attnmlp_r16_contentw2_dec1p5
    export ELF_LORA_RANK=16
    export ELF_LORA_ALPHA=32
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LR=1e-5
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_LOSS_WEIGHT=1.5
    export COND_DROPOUT_PROB=0.0
    ;;
  3)
    variant=fulladapter_frozenelf_contentw2_dec1p5
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_LOSS_WEIGHT=1.5
    export COND_DROPOUT_PROB=0.0
    ;;
  4)
    variant=fulladapter_frozenelf_contentw2_bow0p1_dec1p5
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.1
    export DECODER_LOSS_WEIGHT=1.5
    export COND_DROPOUT_PROB=0.0
    ;;
  5)
    variant=fulladapter_lora_attnmlp_r16_contentw2_bow0p1_dec1p5
    export ELF_LORA_RANK=16
    export ELF_LORA_ALPHA=32
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LR=1e-5
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.1
    export DECODER_LOSS_WEIGHT=1.5
    export COND_DROPOUT_PROB=0.0
    ;;
  6)
    variant=fulladapter_frozenelf_contentw2_bow1p0_dec1p5
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=1.0
    export DECODER_LOSS_WEIGHT=1.5
    export COND_DROPOUT_PROB=0.0
    ;;
  7)
    variant=fulladapter_lora_attnmlp_r16_contentw2_bow1p0_dec1p5
    export ELF_LORA_RANK=16
    export ELF_LORA_ALPHA=32
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LR=1e-5
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=1.0
    export DECODER_LOSS_WEIGHT=1.5
    export COND_DROPOUT_PROB=0.0
    ;;
  *)
    echo "Unsupported array task: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

export RUN_TAG=fmri_minilm_stage2_fullchain_${variant}_from_delaytok_best_ep150_seed49_20260830
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export NPZ_PATH=$known_dir/oracle_exact_minilm384_train_val.npz
export SOURCE_NPZ=$NPZ_PATH
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export TRAIN_BALANCE_KEY=story

export BRAIN_INPUT_NPZ=$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz
export BRAIN_TRAIN_KEY=train_x
export BRAIN_VAL_KEY=val_x
export BRAIN_MODEL_CHECKPOINT=$baseline_checkpoint
export BRAIN_SEMANTIC_OUTPUT_DIM=1536
export BRAIN_HIDDEN_DIM=2048
export BRAIN_RES_BLOCKS=4
export BRAIN_DROPOUT=0.1
export BRAIN_UNFREEZE=projector
export BRAIN_LR=2e-6

export SEMANTIC_ADAPTER_KIND=residual_delay_tokens
export SEMANTIC_MAPPER_HIDDEN_DIM=1024
export SEMANTIC_DELAY_TOKENS=4
export SEMANTIC_DELAY_LAYERS=2
export SEMANTIC_DELAY_HEADS=8
export ADAPTER_DROPOUT=0.1
export SEMANTIC_INPUT_PROJECTION_DIM=0
export TRAIN_SEMANTIC_INPUT_ONLY=0
export NORMALIZE_SEMANTIC_PROJECTION=0
export INIT_E2E_CHECKPOINT=$baseline_checkpoint
export INIT_E2E_ADAPTER_MISMATCH=error

export SEMANTIC_ALIGNMENT_TARGET_NPZ=$known_dir/oracle_exact_minilm384_train_val.npz
export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings
export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=1.0
export SEMANTIC_ALIGNMENT_LOSS_TYPE=cosine
export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.1
export SEMANTIC_CONTRASTIVE_TEMPERATURE=0.07
export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.1

export TARGET_LATENTS_CACHE=$known_dir/oracle_exact_minilm384_trainval_current10_t5small_latents_fp16.npy
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
export EPOCHS=150
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export LR=1e-5
export WEIGHT_DECAY=0.001
export LR_SCHEDULE=cosine
export WARMUP_STEPS=1500
export MIN_LR_RATIO=0.01
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CHECKPOINT_METRIC=content_word_overlap
export SAVE_FINAL_CHECKPOINT=1
export SAVE_EVAL_CHECKPOINTS=1
export EVAL_CHECKPOINT_TOP_K=1
export EVAL_CHECKPOINT_EVERY=0
export SEED=49
export USE_WANDB=1
export WANDB_RUN_GROUP=fmri_minilm_stage2_fullchain_e2e_val_20260830
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="True full-chain raw-fMRI E2E from frozen step-35000 baseline. Variant=$variant. Train MRI projector, ordered delay mapper, complete MiniLM-to-ELF context projector, and optional ELF LoRA for 150 epochs. Optional content BOW coverage loss=$DECODER_CONTENT_BOW_LOSS_WEIGHT. Train11725/val266 only; test107 is never loaded."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
