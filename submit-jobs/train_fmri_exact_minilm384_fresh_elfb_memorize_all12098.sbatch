#!/usr/bin/env bash
#SBATCH --job-name=fmri_txt_mem
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=submit-jobs/train_fmri_exact_minilm384_fresh_elfb_memorize_all12098-%j.out
#SBATCH --error=submit-jobs/train_fmri_exact_minilm384_fresh_elfb_memorize_all12098-%j.err

set -euo pipefail

# Known-text / unseen-brain experiment.  The semantic-to-text model may
# memorize all target sentences through exact MiniLM384 embeddings.  It must
# never see an MRI2SEM-predicted vector during optimization or checkpoint
# selection.  The fixed brain mapper is connected only by the dependent eval.

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1

export NPZ_PATH=$known_dir/oracle_exact_minilm384_all12098_testfirst.npz
export SOURCE_NPZ=$NPZ_PATH
export TARGET_LATENTS_CACHE=$known_dir/fmri_minilm_knowntext_exact384_all12098_testfirst_full_k64_ep100_b64_seed49_20260829_t5small_latents_fp16.npy

export RUN_TAG=${RUN_TAG:-fmri_exact_minilm384_fresh_elfb_full_memorize_all12098_ep600_seed7_20260901}
export OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/$RUN_TAG}

# Fresh stock pretrained ELF-B and a newly initialized semantic adapter.
# Deliberately do not inherit the earlier known-text decoder.
export MODEL=ELF-B
export CHECKPOINT_PATH=embedded-language-flows/ELF-B-owt-torch
export INIT_E2E_CHECKPOINT=
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence

export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=107
export RETRIEVAL_NUM_EXAMPLES=107
export EPOCHS=600
export BATCH_SIZE=64
export EVAL_EVERY=1000
export RETRIEVAL_EVAL_EVERY=1000
export RETRIEVAL_BATCH_SIZE=256

export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_ADAPTER_KIND=flat
export SEMANTIC_INPUT_PROJECTION_DIM=0
export ADAPTER_DROPOUT=0.0
export FREEZE_ELF=0
export LAST_N_BLOCKS=-1
export ELF_LORA_RANK=0

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

# Match the successful Sherlock-SVA full-unfreeze memorization recipe.
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
export WANDB_RUN_GROUP=fmri_knowntext_exact384_fresh_elfb_memorization_20260901
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_SAMPLE_EXAMPLES=16
export WANDB_NOTES="Known-text/unseen-brain control. Fresh stock pretrained ELF-B plus a new exact-MiniLM384-to-K64 adapter; train every ELF-B and adapter parameter on all 12,098 exact semantic/text pairs, including the 107 target texts. No MRI2SEM-predicted vector is loaded during training, evaluation-based checkpoint selection, or sampling. The fixed leakage-safe brain mapper is introduced only after training."
export SEED=7

export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
