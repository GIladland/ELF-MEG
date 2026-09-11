#!/usr/bin/env bash
#SBATCH --job-name=x1best_oofhyb
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-3%3
#SBATCH --output=submit-jobs/train_qc4wyals_x1best_oof5_hybrid_adapter_20260909-%A_%a.out
#SBATCH --error=submit-jobs/train_qc4wyals_x1best_oof5_hybrid_adapter_20260909-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
interface=$DATA_ROOT/tang_themoth_qc4wyals_nested_oof_20260908/interface
old_x1_best=/data/engs-pnpl/glandau/elf-runs/elf_semantic_scaling_meg_ada_x1_trained_adapter_val110_seed49_20260902/best.pt

export SOURCE_NPZ=$interface/qc4wyals_oof_predicted_train_valtail_ada002.npz
export NPZ_PATH=$SOURCE_NPZ
export SEMANTIC_ALIGNMENT_TARGET_NPZ=$interface/qc4wyals_oof_exact_alignment_train_valtail_ada002.npz
export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export MODEL=ELF-B
export CHECKPOINT_PATH=embedded-language-flows/ELF-B-owt-torch
export INIT_E2E_CHECKPOINT=$old_x1_best
export INIT_E2E_ADAPTER_MISMATCH=error
export SEMANTIC_ADAPTER_KIND=flat
export SEMANTIC_INPUT_PROJECTION_DIM=0
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export ADAPTER_DROPOUT=0.05
export NORMALIZE_SEMANTIC_PROJECTION=0
export TRAIN_SEMANTIC_INPUT_ONLY=0
export FREEZE_SEMANTIC_ADAPTER=0
export FREEZE_ELF=1
export LAST_N_BLOCKS=-1
export ELF_LORA_RANK=0
export ELF_LR=0
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=110
export EVAL_NUM_EXAMPLES=110
export RETRIEVAL_NUM_EXAMPLES=110
export GENERATION_T5_RETRIEVAL=1
export BATCH_SIZE=64
export EPOCHS=${HYBRID_EPOCHS:-300}
export EVAL_EVERY=${HYBRID_EVAL_EVERY:-500}
export RETRIEVAL_EVAL_EVERY=${HYBRID_EVAL_EVERY:-500}
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export LR_SCHEDULE=cosine
export WARMUP_STEPS=250
export MIN_LR_RATIO=0.01
export WEIGHT_DECAY=0.0
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_PROB=0.5
export DECODER_NOISE_SCALE=2.5
export DECODER_CONTENT_TOKEN_WEIGHT=2.0
export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.10
export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.05
export DECODER_CONDITION_PAIRING_MARGIN_WEIGHT=0.05
export DECODER_CONDITION_PAIRING_MARGIN=0.10
export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=0.0
export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.0
export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.10
export TRAIN_TARGET_MASK_MODE=full
export CHECKPOINT_METRIC=word_content_mean
export SAVE_FINAL_CHECKPOINT=0
export SAVE_EVAL_CHECKPOINTS=0
export EVAL_CHECKPOINT_TOP_K=1
export EVAL_CHECKPOINT_EVERY=0
export SEED=49

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    arm=lowlr_balanced
    export LR=2e-6
    ;;
  1)
    arm=midlr_balanced
    export LR=5e-6
    ;;
  2)
    arm=lowlr_strongcontent
    export LR=2e-6
    export DECODER_CONTENT_TOKEN_WEIGHT=4.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.25
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.10
    export DECODER_CONDITION_PAIRING_MARGIN_WEIGHT=0.10
    ;;
  3)
    arm=lowlr_strongcontext
    export LR=2e-6
    export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.50
    ;;
  *) exit 2 ;;
esac

for required in "$SOURCE_NPZ" "$SEMANTIC_ALIGNMENT_TARGET_NPZ" "$old_x1_best"; do
  [[ -f "$required" ]] || { echo "Missing required input: $required" >&2; exit 2; }
done

export RUN_TAG=qc4wyals_x1best_oof5_hybrid_${arm}_ep${EPOCHS}_seed49_20260909
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$interface/flat_context_t5small_latents_fp16.npy
export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1
export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=qc4wyals_x1best_oof5_hybrid_20260909
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_SAMPLE_EXAMPLES=20
export WANDB_NOTES="Hybrid repair of the audited x1 high-content ELF-B adapter using 2,652 nested session-OOF MEG predictions. Frozen ELF-B; val110 selects word/content mean; protected test26 absent."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
