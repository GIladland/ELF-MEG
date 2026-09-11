#!/usr/bin/env bash
#SBATCH --job-name=fmri_s16_final
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=submit-jobs/eval_fmri_minilm_residual_sampler16_cfg0p5_locked_test107-%j.out
#SBATCH --error=submit-jobs/eval_fmri_minilm_residual_sampler16_cfg0p5_locked_test107-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
source_run=fmri_minilm_knowntext_oofmix1536_resmlp_h512_norm_lr5e4_inputonly_ep20_b32_seed49_20260829
prior_eval=eval_${source_run}_unseenbrain_pred1536_decodeseed101_test107_20260829

export NPZ_PATH=$DATA_ROOT/minilm/oracle1536_delayed_seg10_middle10content_clockfix/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_pred1536_test_middle10content_clockfix.npz
export SOURCE_NPZ=$NPZ_PATH
export RUN_TAG=eval_${source_run}_LOCKED_VALSELECTED_steps16_cfg0p5_seed101_unseenbrain_test107_20260830
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$known_dir/eval_latents/${prior_eval}_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/$source_run/best.pt
export INIT_E2E_ADAPTER_MISMATCH=error
export SEMANTIC_ALIGNMENT_TARGET_NPZ=$known_dir/oracle_exact_minilm384_test107.npz
export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings

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
export SEMANTIC_DELAY_TOKENS=4
export SEMANTIC_INPUT_PROJECTION_DIM=0
export NORMALIZE_SEMANTIC_PROJECTION=1
export FREEZE_ELF=1
export EVAL_ONLY=1
export NUM_SAMPLING_STEPS=16
export CFG_SCALE=0.5
export SELF_COND_CFG_SCALE=1.0
export SEED=101
export USE_WANDB=1
export WANDB_RUN_GROUP=fmri_minilm_residual_locked_sampler_test107_20260830
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="LOCKED HELD-OUT REPLAY. The residual checkpoint and 16-step/CFG-0.5/seed-101 decoder were fixed from three-seed predicted-val266 results. MRI2SEM and ELF are frozen; no 107-row test vector or label is used for training or selection."
export SAVE_FINAL_CHECKPOINT=0

test -f "$TARGET_LATENTS_CACHE"
exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
