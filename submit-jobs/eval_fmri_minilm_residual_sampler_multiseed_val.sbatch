#!/usr/bin/env bash
#SBATCH --job-name=fmri_res_cfg
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-17%6
#SBATCH --output=submit-jobs/eval_fmri_minilm_residual_sampler_multiseed_val-%A_%a.out
#SBATCH --error=submit-jobs/eval_fmri_minilm_residual_sampler_multiseed_val-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
source_run=${SOURCE_RUN:-fmri_minilm_knowntext_oofmix1536_resmlp_h512_norm_lr5e4_inputonly_ep20_b32_seed49_20260829}

setting=$((SLURM_ARRAY_TASK_ID / 3))
seed_slot=$((SLURM_ARRAY_TASK_ID % 3))

case "$seed_slot" in
  0) seed=7 ;;
  1) seed=23 ;;
  2) seed=101 ;;
  *) exit 2 ;;
esac

case "$setting" in
  0) sampling_steps=16; cfg_scale=0.25 ;;
  1) sampling_steps=16; cfg_scale=0.5 ;;
  2) sampling_steps=32; cfg_scale=0.25 ;;
  3) sampling_steps=32; cfg_scale=0.5 ;;
  4) sampling_steps=64; cfg_scale=0.25 ;;
  5) sampling_steps=64; cfg_scale=0.5 ;;
  *) exit 2 ;;
esac

cfg_tag=${cfg_scale/./p}
export NPZ_PATH=$DATA_ROOT/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz
export SOURCE_NPZ=$NPZ_PATH
export RUN_TAG=eval_${source_run}_predval266_steps${sampling_steps}_cfg${cfg_tag}_seed${seed}_20260830
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$known_dir/oofmix_current10_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/$source_run/best.pt
export INIT_E2E_ADAPTER_MISMATCH=error
export SEMANTIC_ALIGNMENT_TARGET_NPZ=$known_dir/oofmix_exact_minilm384_targets_train23450_val266.npz

export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=266
export EVAL_NUM_EXAMPLES=266
export RETRIEVAL_NUM_EXAMPLES=266
export RETRIEVAL_BATCH_SIZE=256
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_ADAPTER_KIND=residual_mlp
export SEMANTIC_MAPPER_HIDDEN_DIM=512
export SEMANTIC_DELAY_TOKENS=4
export SEMANTIC_INPUT_PROJECTION_DIM=0
export NORMALIZE_SEMANTIC_PROJECTION=1
export FREEZE_ELF=1
export EVAL_ONLY=1
export NUM_SAMPLING_STEPS=$sampling_steps
export CFG_SCALE=$cfg_scale
export SELF_COND_CFG_SCALE=1.0
export SEED=$seed
export USE_WANDB=1
export WANDB_RUN_GROUP=fmri_minilm_residual_sampler_multiseed_val_20260830
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Validation-only sampler robustness sweep for the selected leakage-safe residual mapper. Compares steps and CFG under three predeclared diffusion seeds using predicted val266 only; test107 is absent."
export SAVE_FINAL_CHECKPOINT=0

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
