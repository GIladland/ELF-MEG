#!/usr/bin/env bash
#SBATCH --job-name=fmri_preserve
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-4
#SBATCH --output=submit-jobs/train_fmri_freshmem_semantic_preservation_matrix-%A_%a.out
#SBATCH --error=submit-jobs/train_fmri_freshmem_semantic_preservation_matrix-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
snapshot=${ORACLE_SNAPSHOT:?Set ORACLE_SNAPSHOT}
snapshot_tag=${ORACLE_SNAPSHOT_TAG:-step024000}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1

export NPZ_PATH=$DATA_ROOT/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz
export SOURCE_NPZ=$NPZ_PATH
export SEMANTIC_ALIGNMENT_TARGET_NPZ=$known_dir/oofmix_exact_minilm384_targets_train23450_val266.npz
export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings
export SEMANTIC_ALIGNMENT_LOSS_TYPE=cosine
export SEMANTIC_CONTRASTIVE_TEMPERATURE=0.07
export SEMANTIC_RANK_DISTILL_TEMPERATURE=0.07
export TARGET_LATENTS_CACHE=$known_dir/oofmix_current10_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=$snapshot
export INIT_E2E_ADAPTER_MISMATCH=residual-mlp

export SEMANTIC_ADAPTER_KIND=residual_mlp
export SEMANTIC_MAPPER_HIDDEN_DIM=512
export SEMANTIC_DELAY_TOKENS=4
export TRAIN_SEMANTIC_INPUT_ONLY=1
export ADAPTER_DROPOUT=0.0
export FREEZE_ELF=1
export ELF_LORA_RANK=0
export ELF_LORA_ALPHA=8
export ELF_LORA_DROPOUT=0.0
export ELF_LORA_TARGETS=attention,mlp
export ELF_LORA_LAST_N_BLOCKS=2
export ELF_LR=0

export LR=1e-5
export EPOCHS=${EPOCHS:-12}
export BATCH_SIZE=32
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=266
export EVAL_NUM_EXAMPLES=266
export RETRIEVAL_NUM_EXAMPLES=266
export EVAL_EVERY=500
export RETRIEVAL_EVAL_EVERY=500
export RETRIEVAL_BATCH_SIZE=256
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
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
export DECODER_CONDITION_PAIRING_MARGIN_WEIGHT=0.0
export DECODER_CONDITION_PAIRING_MARGIN=0.1
export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=0.0
export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.0
export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.0
export SEMANTIC_CONTENT_LOSS_WEIGHT=0.0
export SEMANTIC_RAW_ANCHOR_LOSS_WEIGHT=1.0
export SEMANTIC_GEOMETRY_LOSS_WEIGHT=1.0
export SEMANTIC_RANK_DISTILL_LOSS_WEIGHT=0.0
export CURRICULUM_SEMANTIC_ONLY_EPOCHS=2
export CURRICULUM_DENOISER_LOSS_WEIGHT=0.1
export CURRICULUM_DECODER_LOSS_WEIGHT=0.1

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    variant=residual_anchor_geometry
    export SEMANTIC_RAW_ANCHOR_LOSS_WEIGHT=2.0
    export SEMANTIC_GEOMETRY_LOSS_WEIGHT=2.0
    ;;
  1)
    variant=residual_rankdistill_pair
    export SEMANTIC_RANK_DISTILL_LOSS_WEIGHT=1.0
    export DECODER_CONDITION_PAIRING_MARGIN_WEIGHT=0.1
    ;;
  2)
    variant=residual_exactsemantic_rank_pair
    export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=1.0
    export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.1
    export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.1
    export SEMANTIC_RANK_DISTILL_LOSS_WEIGHT=0.5
    export DECODER_CONDITION_PAIRING_MARGIN_WEIGHT=0.25
    ;;
  3)
    variant=residual_exactsemantic_rank_pair_content
    export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=1.0
    export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.1
    export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.1
    export SEMANTIC_RANK_DISTILL_LOSS_WEIGHT=0.5
    export DECODER_CONDITION_PAIRING_MARGIN_WEIGHT=0.25
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.5
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.1
    ;;
  4)
    variant=residual_exactsemantic_rank_pair_content_lora4last2
    export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=1.0
    export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.1
    export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.1
    export SEMANTIC_RANK_DISTILL_LOSS_WEIGHT=0.5
    export DECODER_CONDITION_PAIRING_MARGIN_WEIGHT=0.25
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.5
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.1
    export ELF_LORA_RANK=4
    export ELF_LR=1e-6
    ;;
  *) exit 2 ;;
esac

export RUN_TAG=fmri_freshmem_semantic_preservation_${snapshot_tag}_${variant}_val266_seed49_20260901
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export WEIGHT_DECAY=0.0
export LR_SCHEDULE=cosine
export WARMUP_STEPS=250
export MIN_LR_RATIO=0.02
export CHECKPOINT_METRIC=semantic_top5
export SAVE_FINAL_CHECKPOINT=1
export SAVE_EVAL_CHECKPOINTS=1
export EVAL_CHECKPOINT_TOP_K=5
export EVAL_CHECKPOINT_EVERY=2
export GENERATION_T5_RETRIEVAL=1
export SEED=49

export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=fmri_freshmem_semantic_preservation_20260901
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_SAMPLE_EXAMPLES=16
export WANDB_NOTES="Retrieval-preserving residual MiniLM interface from fresh exact-text-memorized ELF snapshot $snapshot_tag. Variant=$variant. Zero-init residual around raw four-delay mean; frozen oracle context net; optional last-two-block LoRA. Train uses exact oracle plus story-OOF train predictions, selects on val266, and never loads test107."

export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
