#!/usr/bin/env bash
#SBATCH --job-name=ta_ada_oracle
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-5%4
#SBATCH --output=submit-jobs/train_tang_apples_ada2text_oracle_matrix-%A_%a.out
#SBATCH --error=submit-jobs/train_tang_apples_ada2text_oracle_matrix-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
data_dir=$DATA_ROOT/tang_themoth_apples_exact_ada002_textmid_seg10_v1

export SOURCE_NPZ=$data_dir/tang_apples_ada002_knowntext_all2788_testfirst.npz
export NPZ_PATH=$SOURCE_NPZ
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export MODEL=ELF-B
export CHECKPOINT_PATH=embedded-language-flows/ELF-B-owt-torch
export SEMANTIC_ADAPTER_KIND=flat
export SEMANTIC_INPUT_PROJECTION_DIM=0
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export ADAPTER_DROPOUT=0.0
export NORMALIZE_SEMANTIC_PROJECTION=0
export FREEZE_ELF=0
export FREEZE_SEMANTIC_ADAPTER=0
export LAST_N_BLOCKS=-1
export ELF_LORA_RANK=0
export ELF_LORA_ALPHA=16
export ELF_LORA_DROPOUT=0.05
export ELF_LORA_TARGETS=attention,mlp
export ELF_LORA_LAST_N_BLOCKS=-1
export LR=2e-5
export ELF_LR=0
export EPOCHS=300
export BATCH_SIZE=64
export DECODER_CONTENT_TOKEN_WEIGHT=1.0
export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.0
export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.0

case "$SLURM_ARRAY_TASK_ID" in
  0)
    arm=elfb_full_standard
    ;;
  1)
    arm=elfb_full_content
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.10
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.05
    ;;
  2)
    arm=elfb_lora8_content
    export FREEZE_ELF=1
    export ELF_LORA_RANK=8
    export ELF_LORA_ALPHA=16
    export LR=1e-4
    export ELF_LR=2e-5
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.10
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.05
    ;;
  3)
    arm=elfb_last4_content
    export LAST_N_BLOCKS=4
    export LR=3e-5
    export ELF_LR=1e-5
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.10
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.05
    ;;
  4)
    arm=elfb_adapter_only
    export FREEZE_ELF=1
    export LR=1e-4
    ;;
  5)
    arm=elfl_full_content
    export MODEL=ELF-L
    export CHECKPOINT_PATH=embedded-language-flows/ELF-L-owt-torch
    export BATCH_SIZE=32
    export EPOCHS=150
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.10
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.05
    ;;
  *) exit 2 ;;
esac

if [[ -n "${ORACLE_EPOCHS:-}" ]]; then
  export EPOCHS=$ORACLE_EPOCHS
fi
oracle_suffix=${ORACLE_SUFFIX:-}
if [[ -n "$oracle_suffix" ]]; then
  oracle_suffix=_$oracle_suffix
fi
export RUN_TAG=tang_apples_ada002_${arm}_knowntext_all2788${oracle_suffix}_seed49_20260901
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=26
export RETRIEVAL_NUM_EXAMPLES=26
export GENERATION_T5_RETRIEVAL=1
export EVAL_EVERY=${ORACLE_EVAL_EVERY:-1000}
export RETRIEVAL_EVAL_EVERY=${ORACLE_EVAL_EVERY:-1000}
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export LR_SCHEDULE=cosine
export WARMUP_STEPS=500
export MIN_LR_RATIO=0.01
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_PROB=0.5
export DECODER_NOISE_SCALE=2.5
export TRAIN_TARGET_MASK_MODE=full
export CHECKPOINT_METRIC=content_word_overlap
export SAVE_FINAL_CHECKPOINT=1
export SAVE_EVAL_CHECKPOINTS=1
export EVAL_CHECKPOINT_TOP_K=3
export EVAL_CHECKPOINT_EVERY=1
export SEED=49
export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=tang_apples_ada2text_oracle_matrix_20260901
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_SAMPLE_EXAMPLES=16
export TARGET_LATENTS_CACHE=$data_dir/tang_apples_ada002_knowntext_all2788_t5small_latents_fp16.npy
export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1
export WANDB_NOTES="Intentional Tang/Apples known-text ADA-to-ELF overfit. All 2,788 exact ADA/text rows are trainable, including the 26 later MEG evaluation texts. No MEG-predicted vector is loaded in this oracle matrix."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
