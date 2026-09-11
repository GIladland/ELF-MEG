#!/usr/bin/env bash
#SBATCH --job-name=fmri_elf_xattn
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-5
#SBATCH --output=submit-jobs/fmri_elf_brain_cross_attention_val-%A_%a.out
#SBATCH --error=submit-jobs/fmri_elf_brain_cross_attention_val-%A_%a.err

set -euo pipefail

# Explicit target-to-brain cross-attention for ELF diffusion. Development uses
# train11725 and val266 only; the 107 test brain rows are never loaded.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
mri_run=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1
init_checkpoint=${INIT_CHECKPOINT_OVERRIDE:-/data/engs-pnpl/glandau/elf-runs/fmri_minilm_stage2_recover2_lora_last2_r4_semantic_strong_from_delaytok_best_ep100_seed49_20260830_from_lora_step3000/checkpoints/eval_step_00001000_score_0.403006.pt}

task=${SLURM_ARRAY_TASK_ID:-0}
cross_blocks=4
cross_lr=1e-4
pairing_weight=0.0

case "$task" in
  0)
    variant=xattn_last2_lr1e4
    cross_blocks=2
    ;;
  1)
    variant=xattn_last4_lr1e4
    ;;
  2)
    variant=xattn_last4_lr1e4_pair0p1
    pairing_weight=0.1
    ;;
  3)
    variant=xattn_last4_lr3e5_pair0p1
    cross_lr=3e-5
    pairing_weight=0.1
    ;;
  4)
    variant=xattn_last6_lr3e5_pair0p1
    cross_blocks=6
    cross_lr=3e-5
    pairing_weight=0.1
    ;;
  5)
    variant=xattn_last4_lr3e4_pair0p1
    cross_lr=3e-4
    pairing_weight=0.1
    ;;
  *)
    echo "Unsupported task $task" >&2
    exit 2
    ;;
esac

export RUN_TAG=fmri_minilm_elf_${variant}_from_recover2_step1000_20260831${CROSS_RUN_SUFFIX:-}
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export NPZ_PATH=$known_dir/oracle_exact_minilm384_train_val.npz
export SOURCE_NPZ=$NPZ_PATH
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export TRAIN_BALANCE_KEY=story

export BRAIN_INPUT_NPZ=$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz
export BRAIN_TRAIN_KEY=train_x
export BRAIN_VAL_KEY=val_x
export BRAIN_MODEL_CHECKPOINT=$init_checkpoint
export BRAIN_SEMANTIC_OUTPUT_DIM=1536
export BRAIN_PROJECTOR_MISMATCH=error
export BRAIN_HIDDEN_DIM=2048
export BRAIN_RES_BLOCKS=4
export BRAIN_DROPOUT=0.1
export BRAIN_UNFREEZE=frozen
export BRAIN_LR=0

export SEMANTIC_ADAPTER_KIND=residual_delay_tokens
export SEMANTIC_MAPPER_HIDDEN_DIM=1024
export SEMANTIC_DELAY_TOKENS=4
export SEMANTIC_DELAY_LAYERS=2
export SEMANTIC_DELAY_HEADS=8
export ADAPTER_DROPOUT=0.1
export SEMANTIC_INPUT_PROJECTION_DIM=0
export TRAIN_SEMANTIC_INPUT_ONLY=0
export FREEZE_SEMANTIC_ADAPTER=1
export NORMALIZE_SEMANTIC_PROJECTION=0
export SEMANTIC_PRESERVE_DELAY_CONTEXT=0
export INIT_E2E_CHECKPOINT=$init_checkpoint
export INIT_E2E_ADAPTER_MISMATCH=error

# Recreate the preserved rank-4 modules for strict checkpoint loading, then
# freeze every ELF tensor except the new zero-init cross-attention adapters.
export FREEZE_ELF=1
export ELF_LORA_RANK=4
export ELF_LORA_ALPHA=8
# The imported LoRA path is frozen in this stage, so use its evaluation-time
# behavior rather than injecting dropout into a fixed baseline.
export ELF_LORA_DROPOUT=0.0
export ELF_LORA_TARGETS=attention,mlp
export ELF_LORA_LAST_N_BLOCKS=2
export ELF_BRAIN_CROSS_ATTENTION_LAST_N_BLOCKS=$cross_blocks
export ELF_BRAIN_CROSS_ATTENTION_HEADS=12
export ELF_BRAIN_CROSS_ATTENTION_DROPOUT=0.0
export ELF_BRAIN_CROSS_ATTENTION_ONLY=1

export TARGET_LATENTS_CACHE=$known_dir/oracle_exact_minilm384_trainval_current10_t5small_latents_fp16.npy
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export SKIP_ARC_SETUP=1
export NUM_EXAMPLES=0
# This value defines the physical train/validation boundary in the packed NPZ,
# not merely evaluation cost.  Keep it immutable so a smoke-test override can
# never move held-out rows into training.
if [[ "${VAL_NUM_EXAMPLES_OVERRIDE:-266}" != "266" ]]; then
  echo "VAL_NUM_EXAMPLES_OVERRIDE cannot change the locked 266-row validation boundary" >&2
  exit 2
fi
export VAL_NUM_EXAMPLES=266
export EVAL_NUM_EXAMPLES=${EVAL_NUM_EXAMPLES_OVERRIDE:-266}
export RETRIEVAL_NUM_EXAMPLES=${RETRIEVAL_NUM_EXAMPLES_OVERRIDE:-266}
export RETRIEVAL_EVAL_EVERY=${RETRIEVAL_EVAL_EVERY_OVERRIDE:-250}
export RETRIEVAL_BATCH_SIZE=256
export EPOCHS=${EPOCHS_OVERRIDE:-20}
export BATCH_SIZE=32
export EVAL_EVERY=${EVAL_EVERY_OVERRIDE:-250}
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export LR=1e-4
export ELF_LR=$cross_lr
export WEIGHT_DECAY=0.001
export LR_SCHEDULE=cosine
export WARMUP_STEPS=250
export MIN_LR_RATIO=0.05
export COND_DROPOUT_PROB=0.0
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32

# Decoder-only cross-attention deliberately leaves ELF's pretrained flow path
# untouched.  Wrappers can therefore disable a denoiser objective that has no
# trainable route without changing the original shared-flow screen default.
export DENOISER_LOSS_WEIGHT=${DENOISER_LOSS_WEIGHT_OVERRIDE:-1.0}
export DECODER_LOSS_WEIGHT=1.0
export DECODER_CONTENT_TOKEN_WEIGHT=${DECODER_CONTENT_TOKEN_WEIGHT_OVERRIDE:-2.0}
export DECODER_CONTENT_BOW_LOSS_WEIGHT=${DECODER_CONTENT_BOW_LOSS_WEIGHT_OVERRIDE:-0.0}
export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=${DECODER_CONTENT_PRECISION_LOSS_WEIGHT_OVERRIDE:-0.0}
export DECODER_CONDITION_PAIRING_MARGIN_WEIGHT=$pairing_weight
export DECODER_CONDITION_PAIRING_MARGIN=0.1

export CHECKPOINT_METRIC=content_word_overlap
export SAVE_FINAL_CHECKPOINT=1
export SAVE_EVAL_CHECKPOINTS=1
export EVAL_CHECKPOINT_TOP_K=2
export EVAL_CHECKPOINT_EVERY=0
export SEED=49
export USE_WANDB=${USE_WANDB_OVERRIDE:-1}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-fmri_elf_brain_cross_attention_val_20260831}
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES=${WANDB_NOTES:-"Leakage-safe raw-fMRI ELF diffusion cross-attention screen. Frozen recover2 ELF/LoRA and frozen MRI2SEM/context adapter; train only zero-init cross-attention in final $cross_blocks blocks at LR $cross_lr. Pairing weight=$pairing_weight. Train11725/val266 only; test107 never loaded. Promotion target: content F1 > 0.043743 with matched brain controls."}

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
