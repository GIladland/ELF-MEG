#!/usr/bin/env bash
#SBATCH --job-name=fmri_txt_eval
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-1
#SBATCH --output=submit-jobs/eval_fmri_exact_minilm384_fresh_elfb_memorize_test107-%A_%a.out
#SBATCH --error=submit-jobs/eval_fmri_exact_minilm384_fresh_elfb_memorize_test107-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
train_tag=${TRAIN_TAG:-fmri_exact_minilm384_fresh_elfb_full_memorize_all12098_ep600_seed7_20260901}

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    condition=exact_minilm384_oracle
    export NPZ_PATH=$known_dir/oracle_exact_minilm384_test107.npz
    projection_dim=0
    adapter_mismatch=error
    ;;
  1)
    condition=unseen_fmri_delayed1536_meanblocks
    source_dir=$DATA_ROOT/minilm/oracle1536_delayed_seg10_middle10content_clockfix
    export NPZ_PATH=$source_dir/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_pred1536_test_middle10content_clockfix.npz
    projection_dim=384
    adapter_mismatch=mean-blocks
    ;;
  *) exit 2 ;;
esac

export SOURCE_NPZ=$NPZ_PATH
export RUN_TAG=eval_${train_tag}_${condition}_seed7_20260901
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$known_dir/eval_latents/${RUN_TAG}_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=${INIT_E2E_CHECKPOINT_OVERRIDE:-/data/engs-pnpl/glandau/elf-runs/$train_tag/best.pt}
export INIT_E2E_ADAPTER_MISMATCH=$adapter_mismatch

export MODEL=ELF-B
export CHECKPOINT_PATH=embedded-language-flows/ELF-B-owt-torch
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export EVAL_ONLY=1
export EPOCHS=0
export NUM_EXAMPLES=107
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=107
export RETRIEVAL_NUM_EXAMPLES=107
export BATCH_SIZE=32
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_ADAPTER_KIND=flat
export SEMANTIC_INPUT_PROJECTION_DIM=$projection_dim
export FREEZE_ELF=1
export LAST_N_BLOCKS=-1
export TRAIN_TARGET_MASK_MODE=full
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CHECKPOINT_METRIC=negative_wer

export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=fmri_knowntext_exact384_fresh_elfb_memorization_20260901
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Frozen evaluation of the fresh exact-MiniLM384 known-text decoder on $condition. Task 1 is the only place the 107 held-out MRI2SEM predictions enter; they were absent from diffusion training and checkpoint selection."
export SEED=7

export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
