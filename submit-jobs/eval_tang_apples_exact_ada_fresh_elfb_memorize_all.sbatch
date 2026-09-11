#!/usr/bin/env bash
#SBATCH --job-name=ta_ada_mem_eval
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=submit-jobs/eval_tang_apples_exact_ada_fresh_elfb_memorize_all-%j.out
#SBATCH --error=submit-jobs/eval_tang_apples_exact_ada_fresh_elfb_memorize_all-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}

train_tag=${TRAIN_TAG:-tang_apples_exact_ada002_fresh_elfb_full_memorize_all2788_ep1000_seed7_20260901}
data_dir=$DATA_ROOT/tang_themoth_apples_exact_ada002_textmid_seg10_v1
export NPZ_PATH=$data_dir/tang_apples_ada002_knowntext_all2788_testfirst.npz
export TARGET_LATENTS_CACHE=$data_dir/tang_apples_ada002_knowntext_all2788_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=${SOURCE_CHECKPOINT:-/data/engs-pnpl/glandau/elf-runs/$train_tag/best.pt}
if [[ ! -f "$INIT_E2E_CHECKPOINT" ]]; then
  echo "Missing exact-ADA checkpoint: $INIT_E2E_CHECKPOINT" >&2
  exit 2
fi

checkpoint_tag=$(basename "${INIT_E2E_CHECKPOINT%.pt}")
eval_suffix=${EVAL_TAG_SUFFIX:-}
if [[ -n "$eval_suffix" ]]; then
  eval_suffix=_$eval_suffix
fi
export RUN_TAG=${RUN_TAG:-eval_${train_tag}_${checkpoint_tag}${eval_suffix}_all2788}
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export MODEL=ELF-B
export CHECKPOINT_PATH=embedded-language-flows/ELF-B-owt-torch
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence

export EVAL_ONLY=1
export EPOCHS=0
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=0
export RETRIEVAL_NUM_EXAMPLES=-1
export GENERATION_T5_RETRIEVAL=0
export BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
export NUM_SAMPLING_STEPS=32
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_ADAPTER_KIND=flat
export FREEZE_ELF=1
export LAST_N_BLOCKS=-1
export TRAIN_TARGET_MASK_MODE=full

export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=tang_apples_exact_ada_fresh_elfb_memorization_20260901
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Automatic all-2,788 known-text reconstruction audit of the WER-selected fresh ELF-B exact-ADA memorization checkpoint."
export SEED=7

export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
