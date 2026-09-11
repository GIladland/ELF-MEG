#!/usr/bin/env bash
#SBATCH --job-name=qc4_ada_robust
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-2%3
#SBATCH --output=submit-jobs/train_tang_themoth_qc4wyals_ada2text_robust-%A_%a.out
#SBATCH --error=submit-jobs/train_tang_themoth_qc4wyals_ada2text_robust-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
data_dir=$DATA_ROOT/tang_themoth_qc4wyals_ada_v1
exact_npz=$data_dir/exact_ada002_birthofanation26.npz
noise_npz=$data_dir/exact_ada002_birthofanation26_tangentnoise_v1.npz
oracle_run=/data/engs-pnpl/glandau/elf-runs/fmri_knowntext_ada002_elfb_long_ep300_testfirst_k64_seed49_20260830

export FREEZE_ELF=1
export LAST_N_BLOCKS=-1
export ELF_LORA_RANK=0
export ELF_LORA_ALPHA=8
export ELF_LORA_DROPOUT=0.0
export ELF_LORA_TARGETS=attention,mlp
export ELF_LORA_LAST_N_BLOCKS=4
export LR=2e-5
export ELF_LR=0
export EPOCHS=75
export DECODER_CONTENT_TOKEN_WEIGHT=2.0
export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.10
export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.05
export SOURCE_NPZ=$noise_npz

case "$SLURM_ARRAY_TASK_ID" in
  0)
    # Maximum exact-text memorization capacity: update all ELF blocks at a very
    # low rate while adapting the semantic interface more quickly.
    arm=full_elf_exact
    export SOURCE_NPZ=$exact_npz
    export FREEZE_ELF=0
    export LAST_N_BLOCKS=-1
    export LR=1e-5
    export ELF_LR=1e-6
    export EPOCHS=1500
    ;;
  1)
    # Preserve the pretrained diffusion model and robustify only its ADA bridge.
    arm=noise_adapter
    ;;
  2)
    # Let a small diffusion-side LoRA absorb the local ADA neighborhood shift.
    arm=noise_lora4
    export ELF_LORA_RANK=4
    export ELF_LR=3e-6
    ;;
  *) exit 2 ;;
esac

export RUN_TAG=tang_themoth_birthofanation26_ada002_${arm}_seed49_20260901
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export NPZ_PATH=$SOURCE_NPZ
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export MODEL=ELF-B
export CHECKPOINT_PATH=embedded-language-flows/ELF-B-owt-torch
export INIT_E2E_CHECKPOINT=$oracle_run/best.pt
export INIT_E2E_ADAPTER_MISMATCH=error
export SEMANTIC_ADAPTER_KIND=flat
export SEMANTIC_INPUT_PROJECTION_DIM=0
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export ADAPTER_DROPOUT=0.0
export FREEZE_SEMANTIC_ADAPTER=0
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=26
export RETRIEVAL_NUM_EXAMPLES=26
export GENERATION_T5_RETRIEVAL=1
export BATCH_SIZE=16
export EVAL_EVERY=250
export RETRIEVAL_EVAL_EVERY=250
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export LR_SCHEDULE=cosine
export WARMUP_STEPS=100
export MIN_LR_RATIO=0.02
export DECODER_LOSS_WEIGHT=1.0
export DENOISER_LOSS_WEIGHT=1.0
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
export WANDB_RUN_GROUP=tang_themoth_qc4wyals_ada2text_robust_20260901
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_SAMPLE_EXAMPLES=16
export TARGET_LATENTS_CACHE=$data_dir/${arm}_t5small_latents_fp16.npy
export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1
export WANDB_NOTES="Birth-of-a-Nation ADA-to-ELF capacity experiment. Exact-text overfit is intentional. qc4wyals MEG predictions remain evaluation-only. Noisy arms use only synthetic tangent perturbations of exact ADA vectors."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
