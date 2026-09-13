#!/usr/bin/env bash
#SBATCH --job-name=qc4_ooflora
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=submit-jobs/train_qc4wyals_elfm_nested_oof_lora_20260912-%j.out
#SBATCH --error=submit-jobs/train_qc4wyals_elfm_nested_oof_lora_20260912-%j.err

set -euo pipefail

# Distribution-matched pure flow adaptation. This uses nested session-OOF
# predicted ADA for training and deployment predicted ADA for val110. It does
# not use a sentence proposal, lexical classifier, D'Ascoli, or T5 generation.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
interface=$DATA_ROOT/tang_themoth_qc4wyals_nested_oof_20260908/interface
runs_root=/data/engs-pnpl/glandau/elf-runs
oracle=$runs_root/elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906/best.pt
semantic_winner=$runs_root/meg_elfm_ctxdistill_robustinit_exactteacher_outputprojector_ep100_seed49_20260906/best_adapter.pt

export SOURCE_NPZ=$interface/qc4wyals_oof_predicted_train_valtail_ada002.npz
export NPZ_PATH=$SOURCE_NPZ
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export MODEL=ELF-M
export CHECKPOINT_PATH=$oracle
export INIT_E2E_CHECKPOINT=$semantic_winner
export INIT_E2E_ADAPTER_MISMATCH=error
export SEMANTIC_ADAPTER_KIND=flat
export SEMANTIC_HIDDEN_DIM=4096
export CONTEXT_LENGTH=64
export ADAPTER_DROPOUT=0
export FREEZE_SEMANTIC_ADAPTER=1

export FREEZE_ELF=1
export ELF_LORA_RANK=4
export ELF_LORA_ALPHA=16
export ELF_LORA_DROPOUT=0.05
export ELF_LORA_TARGETS=attention
export ELF_LORA_LAST_N_BLOCKS=4
export ELF_LR=3e-5
export LR=3e-5
export WEIGHT_DECAY=0.001
export LR_SCHEDULE=cosine
export WARMUP_STEPS=100
export MIN_LR_RATIO=0.05

export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=110
export EVAL_NUM_EXAMPLES=110
export RETRIEVAL_NUM_EXAMPLES=110
export BATCH_SIZE=32
export EPOCHS=10
export EVAL_EVERY=250
export RETRIEVAL_EVAL_EVERY=250
export RETRIEVAL_BATCH_SIZE=64
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=0.25
export DECODER_PROB=0.5
export DECODER_P_MEAN=-1.5
export DECODER_P_STD=0.8
export DECODER_CONTENT_TOKEN_WEIGHT=2.0
export DECODER_CONDITION_PAIRING_MARGIN_WEIGHT=0.05
export DECODER_CONDITION_PAIRING_MARGIN=0.10
export TRAIN_TARGET_MASK_MODE=full
export CFG_SCALE=1.0
export SELF_COND_CFG_SCALE=1.0

export GENERATION_T5_RETRIEVAL=1
export CHECKPOINT_METRIC=word_content_mean
export SAVE_TRAINABLE_ONLY_CHECKPOINT=1
export SAVE_FINAL_CHECKPOINT=1
export SAVE_EVAL_CHECKPOINTS=0
export USE_WANDB=0
export SEED=49
export RUN_TAG=meg_elfm_puresem_nestedoof_lora4last4_flowjoint_ep10_seed49_20260912
export OUTPUT_DIR=$runs_root/$RUN_TAG
export TARGET_LATENTS_CACHE=$interface/flat_context_t5small_latents_fp16.npy
export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1
export WANDB_NOTES="Nested-OOF predicted-ADA-only ELF-M final-four-block LoRA. Noise start; no D'Ascoli or generated T5 source; val110 only; test26 absent."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
