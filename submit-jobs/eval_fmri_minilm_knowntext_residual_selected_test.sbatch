#!/usr/bin/env bash
#SBATCH --job-name=eval_fmri_res
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --array=0-1
#SBATCH --output=submit-jobs/eval_fmri_minilm_knowntext_residual_selected_test-%A_%a.out
#SBATCH --error=submit-jobs/eval_fmri_minilm_knowntext_residual_selected_test-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
source_dir=$DATA_ROOT/minilm/oracle1536_delayed_seg10_middle10content_clockfix
selected_run=fmri_minilm_knowntext_oofmix1536_resmlp_h512_norm_lr5e4_inputonly_ep20_b32_seed49_20260829

case "$SLURM_ARRAY_TASK_ID" in
  0)
    condition=unseenbrain_pred1536
    export NPZ_PATH=$source_dir/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_pred1536_test_middle10content_clockfix.npz
    ;;
  1)
    condition=oracle_delayed1536
    export NPZ_PATH=$source_dir/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_oracle1536_middle10content_clockfix_test.npz
    ;;
  *) exit 2 ;;
esac

export SOURCE_NPZ=$NPZ_PATH
export RUN_TAG=eval_${selected_run}_${condition}_decodeseed101_test107_20260829
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$known_dir/eval_latents/${RUN_TAG}_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/$selected_run/best.pt
export INIT_E2E_ADAPTER_MISMATCH=error
export SEMANTIC_ALIGNMENT_TARGET_NPZ=$known_dir/oracle_exact_minilm384_test107.npz
export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings

export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export NUM_EXAMPLES=107
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=107
export RETRIEVAL_NUM_EXAMPLES=107
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
export SEED=101
export USE_WANDB=1
export WANDB_RUN_GROUP=fmri_minilm_knowntext_residual_selected_test_20260829
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Single paired held-out evaluation after validation-only model and decoding-seed selection. The residual mapper trains on oracle plus story-OOF predictions and predicted val266 only; test107 brain vectors are unseen."
export SAVE_FINAL_CHECKPOINT=0

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
