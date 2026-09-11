#!/usr/bin/env bash
#SBATCH --job-name=elfm_x16_lora
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/train_elf_archscale_meg_x16_elfm_lora-%j.out
#SBATCH --error=submit-jobs/train_elf_archscale_meg_x16_elfm_lora-%j.err

set -euo pipefail

# Controlled diffusion-side adaptation.  Start from the stable x16 ELF-M
# adapter checkpoint, freeze that adapter and all base ELF weights, and train
# only rank-8 LoRA.  No raw MEG or protected test26 vector is loaded here.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
interface=$DATA_ROOT/tang_themoth_qc4wyals_ada_v2/interface

export NPZ_PATH=$interface/qc4wyals_robust_exactall_predtrain_residualaug_valtail_ada002.npz
export SOURCE_NPZ=$NPZ_PATH
export SEMANTIC_ALIGNMENT_TARGET_NPZ=$interface/qc4wyals_robust_exact_alignment_valtail_ada002.npz
export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings
export TARGET_LATENTS_CACHE=$interface/robust_residualaug_adapter_t5small_latents_fp16.npy

export RUN_TAG=elf_archscale_meg_ada_x16_elfm_frozenadapter_lora8_content_val110_seed49_20260906
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/elf_archscale_meg_ada_x16_elfm_trained_adapter_val110_seed49_20260904/best.pt
export INIT_E2E_ADAPTER_MISMATCH=error
export MODEL=ELF-M
export CHECKPOINT_PATH=embedded-language-flows/ELF-M-owt-torch
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence

export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=110
export EVAL_NUM_EXAMPLES=110
export RETRIEVAL_NUM_EXAMPLES=110
export BATCH_SIZE=16
export EPOCHS=20
export EVAL_EVERY=250
export RETRIEVAL_EVAL_EVERY=250
export RETRIEVAL_BATCH_SIZE=128
export CONTEXT_LENGTH=64
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32

export SEMANTIC_ADAPTER_KIND=flat
export SEMANTIC_HIDDEN_DIM=4096
export ADAPTER_DROPOUT=0.05
export FREEZE_SEMANTIC_ADAPTER=1
export NORMALIZE_SEMANTIC_PROJECTION=0
export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=0.0
export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.0
export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.0

export FREEZE_ELF=1
export ELF_LORA_RANK=8
export ELF_LORA_ALPHA=16
export ELF_LORA_DROPOUT=0.05
export ELF_LORA_TARGETS=attention,mlp
export ELF_LORA_LAST_N_BLOCKS=-1
export LR=1e-6
export ELF_LR=1e-6
export LR_SCHEDULE=cosine
export WARMUP_STEPS=250
export MIN_LR_RATIO=0.01
export WEIGHT_DECAY=0.0

export TRAIN_TARGET_MASK_MODE=full
export CFG_SCALE=1.0
export SELF_COND_CFG_SCALE=1.0
export COND_DROPOUT_PROB=0.0
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_CONTENT_TOKEN_WEIGHT=4.0
export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.25
export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.10
export DECODER_CONDITION_PAIRING_MARGIN_WEIGHT=0.05
export DECODER_CONDITION_PAIRING_MARGIN=0.10
export GENERATION_T5_RETRIEVAL=1
export CHECKPOINT_METRIC=word_content_mean
export SAVE_FINAL_CHECKPOINT=0
export SAVE_EVAL_CHECKPOINTS=0
export SEED=49

export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=elf_archscale_meg_x16_lora_e2e_20260906
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_SAMPLE_EXAMPLES=16
export WANDB_NOTES="ELF-M x16 diffusion-side LoRA test. Stable trained semantic adapter frozen; base ELF frozen; rank-8 attention+MLP LoRA only. Fit excludes val110 and protected test26 brain predictions."
export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
