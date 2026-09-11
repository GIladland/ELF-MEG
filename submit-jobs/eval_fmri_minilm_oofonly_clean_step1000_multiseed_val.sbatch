#!/usr/bin/env bash
#SBATCH --job-name=oofclean_seed
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-2
#SBATCH --output=submit-jobs/eval_fmri_minilm_oofonly_clean_step1000_multiseed_val-%A_%a.out
#SBATCH --error=submit-jobs/eval_fmri_minilm_oofonly_clean_step1000_multiseed_val-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
source_run=fmri_minilm_knowntext_oofonly_CLEAN_h512_norm_lr1e4_inputonly_ep10_b32_seed49_20260830

case "$SLURM_ARRAY_TASK_ID" in
  0) seed=7 ;;
  1) seed=23 ;;
  2) seed=101 ;;
  *) exit 2 ;;
esac

export NPZ_PATH=$DATA_ROOT/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz
export SOURCE_NPZ=$NPZ_PATH
export RUN_TAG=eval_${source_run}_STEP1000_predval266_steps32_decodeseed${seed}_20260830
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$known_dir/oofmix_current10_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/$source_run/checkpoint_step001000_wer0.972180.pt
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
export NUM_SAMPLING_STEPS=32
export SEED=$seed
export USE_WANDB=1
export WANDB_RUN_GROUP=fmri_minilm_oofonly_CLEAN_step1000_multiseed_val_20260830
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Validation-only decoding-seed robustness for the immutable clean OOF-only step-1000 checkpoint. Predicted val266 only; test107 is absent."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
