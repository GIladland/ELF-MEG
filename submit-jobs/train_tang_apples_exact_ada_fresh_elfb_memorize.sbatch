#!/usr/bin/env bash
#SBATCH --job-name=ta_ada_mem
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=submit-jobs/train_tang_apples_exact_ada_fresh_elfb_memorize-%j.out
#SBATCH --error=submit-jobs/train_tang_apples_exact_ada_fresh_elfb_memorize-%j.err

set -euo pipefail

# Deliberate known-text memorization control.  Start from the stock pretrained
# ELF-B checkpoint (not any prior Tang/Apples checkpoint), initialize a fresh
# ADA semantic adapter, and update every ELF-B parameter on all 2,788 exact
# ADA/text pairs.  The first 136 rows are the known 26 test + 110 validation
# texts and are part of the training corpus.

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}

data_dir=$DATA_ROOT/tang_themoth_apples_exact_ada002_textmid_seg10_v1
export NPZ_PATH=${NPZ_PATH:-$data_dir/tang_apples_ada002_knowntext_all2788_testfirst.npz}
export TARGET_LATENTS_CACHE=${TARGET_LATENTS_CACHE:-$data_dir/tang_apples_ada002_knowntext_all2788_t5small_latents_fp16.npy}

export RUN_TAG=${RUN_TAG:-tang_apples_exact_ada002_fresh_elfb_full_memorize_all2788_ep1000_seed7_20260901}
export OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/$RUN_TAG}

export MODEL=ELF-B
export CHECKPOINT_PATH=embedded-language-flows/ELF-B-owt-torch
export INIT_E2E_CHECKPOINT=
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence

export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=136
export RETRIEVAL_NUM_EXAMPLES=136
export EPOCHS=1000
export BATCH_SIZE=64
export EVAL_EVERY=1000
export RETRIEVAL_EVAL_EVERY=1000

export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_ADAPTER_KIND=flat
export ADAPTER_DROPOUT=0.0
export FREEZE_ELF=0
export LAST_N_BLOCKS=-1
export ELF_LORA_RANK=0

# Match the successful Sherlock-SVA full-mask memorization contract.
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

export LR=1e-4
export ELF_LR=1e-4
export WEIGHT_DECAY=0.0
export LR_SCHEDULE=constant
export WARMUP_STEPS=0

export CHECKPOINT_METRIC=negative_wer
export SAVE_FINAL_CHECKPOINT=1
export SAVE_EVAL_CHECKPOINTS=1
export EVAL_CHECKPOINT_TOP_K=12
export EVAL_CHECKPOINT_EVERY=1

export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=tang_apples_exact_ada_fresh_elfb_memorization_20260901
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_SAMPLE_EXAMPLES=16
export WANDB_NOTES="Known-text exact-ADA memorization control. Fresh stock pretrained ELF-B plus a newly initialized 1536-to-K64 adapter; all ELF-B and adapter parameters train on all 2,788 ten-word Tang/TheMoth/Apples fragments, including the 136 audit rows. No MEG/fMRI vector and no prior Tang checkpoint are loaded. Extended Sherlock-SVA full-mask recipe; select by lowest audit WER and do not stop early."
export SEED=7

export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
