#!/usr/bin/env bash
#SBATCH --job-name=fmri_final107
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/eval_fmri_minilm_balanced_oof_final_test107-%j.out
#SBATCH --error=submit-jobs/eval_fmri_minilm_balanced_oof_final_test107-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
source_run=fmri_minilm_knowntext_oofonly_h512_lr1e4_balanced_story_inputonly_ep20_b32_seed49_20260829
baseline_eval=eval_fmri_minilm_knowntext_oofmix1536_adapteronly_ep10_b32_seed49_20260829_unseenbrain_direct1536_adapteronly_test107_20260829

export NPZ_PATH=$DATA_ROOT/minilm/oracle1536_delayed_seg10_middle10content_clockfix/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_pred1536_test_middle10content_clockfix.npz
export SOURCE_NPZ=$NPZ_PATH
export RUN_TAG=eval_${source_run}_locked_steps32_seed49_unseenbrain_test107_20260830
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$DATA_ROOT/minilm/knowntext_unseenbrain_v1/eval_latents/${baseline_eval}_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/$source_run/best.pt
export INIT_E2E_ADAPTER_MISMATCH=error

export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export NUM_EXAMPLES=107
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=107
export RETRIEVAL_NUM_EXAMPLES=107
export RETRIEVAL_BATCH_SIZE=128
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_ADAPTER_KIND=residual_mlp
export SEMANTIC_MAPPER_HIDDEN_DIM=512
export SEMANTIC_INPUT_PROJECTION_DIM=0
export NORMALIZE_SEMANTIC_PROJECTION=0
export FREEZE_ELF=1
export EVAL_ONLY=1
export NUM_SAMPLING_STEPS=32
export SEED=49
export USE_WANDB=1
export WANDB_RUN_GROUP=fmri_minilm_locked_final_test107_20260830
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="LOCKED FINAL TEST. Architecture/checkpoint/steps/seed selected only from OOF train plus predicted val266. This job opens the 107 MRI2SEM test vectors once for final reporting; no training or selection occurs."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
