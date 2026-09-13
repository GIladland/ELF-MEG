#!/usr/bin/env bash
#SBATCH --job-name=qc4_elfm_xa
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-3%2
#SBATCH --output=submit-jobs/train_qc4wyals_elfm_semantic_xattn_20260912-%A_%a.out
#SBATCH --error=submit-jobs/train_qc4wyals_elfm_semantic_xattn_20260912-%A_%a.err

set -euo pipefail

# Pure semantic-conditioned diffusion screen. Inference starts from noise and
# receives only the frozen qc4wyals ADA prediction. D'Ascoli outputs and
# T5-generated sentences/prefixes are absent. The internal frozen t5-small
# latent/tokenizer remains part of ELF's native representation.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
interface=$DATA_ROOT/tang_themoth_qc4wyals_ada_v2/interface
runs_root=/data/engs-pnpl/glandau/elf-runs

oracle=$runs_root/elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906/best.pt
semantic_winner=$runs_root/meg_elfm_ctxdistill_robustinit_exactteacher_outputprojector_ep100_seed49_20260906/best_adapter.pt

task=${SLURM_ARRAY_TASK_ID:-0}
cross_blocks=2
cross_lr=1e-4
content_weight=1.0
pairing_weight=0.0
decoder_p_mean=-0.5
case "$task" in
  0)
    arm=last2_lr1e4_pm0p5_ce1
    ;;
  1)
    arm=last4_lr1e4_pm0p5_ce1
    cross_blocks=4
    ;;
  2)
    arm=last2_lr3e5_pm1p5_ce1
    cross_lr=3e-5
    decoder_p_mean=-1.5
    ;;
  3)
    arm=last2_lr1e4_pm0p5_ce2_pair0p05
    content_weight=2.0
    pairing_weight=0.05
    ;;
  *)
    echo "Unsupported task $task" >&2
    exit 2
    ;;
esac

source_npz=${SOURCE_NPZ_OVERRIDE:-$interface/qc4wyals_predicted_train_valtail_ada002.npz}
source_label=${SOURCE_LABEL:-insample}
if [[ ! "$source_label" =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "SOURCE_LABEL must contain only letters, digits, and underscores" >&2
  exit 2
fi
for required in "$source_npz" "$oracle" "$semantic_winner"; do
  [[ -f "$required" ]] || { echo "Missing required input: $required" >&2; exit 2; }
done

export NPZ_PATH=$source_npz
export SOURCE_NPZ=$source_npz
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export MODEL=ELF-M
# Load the exact-ADA-trained ELF-M generator here, then recover the selected
# qc4wyals semantic projector through INIT_E2E_CHECKPOINT below.
export CHECKPOINT_PATH=$oracle
export INIT_E2E_CHECKPOINT=$semantic_winner
export INIT_E2E_ADAPTER_MISMATCH=error

export SEMANTIC_ADAPTER_KIND=flat
export SEMANTIC_HIDDEN_DIM=4096
export CONTEXT_LENGTH=64
export ADAPTER_DROPOUT=0.0
export FREEZE_SEMANTIC_ADAPTER=1
export NORMALIZE_SEMANTIC_PROJECTION=0

# Preserve the pretrained flow exactly and train only explicit decoder-side
# target-to-semantic cross-attention. Zero initialization makes step zero
# identical to the selected pure-semantic winner.
export FREEZE_ELF=1
export ELF_LORA_RANK=0
export ELF_BRAIN_CROSS_ATTENTION_LAST_N_BLOCKS=$cross_blocks
export ELF_BRAIN_CROSS_ATTENTION_HEADS=16
export ELF_BRAIN_CROSS_ATTENTION_DROPOUT=0.0
export ELF_BRAIN_CROSS_ATTENTION_ONLY=1
export ELF_BRAIN_CROSS_ATTENTION_DECODER_ONLY=1
export LR=1e-4
export ELF_LR=$cross_lr
export WEIGHT_DECAY=0.001
export LR_SCHEDULE=cosine
export WARMUP_STEPS=100
export MIN_LR_RATIO=0.05

export NUM_EXAMPLES=0
# This is the physical tail split, not merely an evaluation-size knob.
if [[ "${VAL_NUM_EXAMPLES_OVERRIDE:-110}" != "110" ]]; then
  echo "VAL_NUM_EXAMPLES_OVERRIDE cannot change the locked val110 boundary" >&2
  exit 2
fi
export VAL_NUM_EXAMPLES=110
export EVAL_NUM_EXAMPLES=${EVAL_NUM_EXAMPLES_OVERRIDE:-110}
export RETRIEVAL_NUM_EXAMPLES=${RETRIEVAL_NUM_EXAMPLES_OVERRIDE:-110}
export EPOCHS=${EPOCHS_OVERRIDE:-20}
export BATCH_SIZE=${BATCH_SIZE_OVERRIDE:-16}
export EVAL_EVERY=${EVAL_EVERY_OVERRIDE:-250}
export RETRIEVAL_EVAL_EVERY=${RETRIEVAL_EVAL_EVERY_OVERRIDE:-250}
export RETRIEVAL_BATCH_SIZE=64
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32

export TRAIN_TARGET_MASK_MODE=full
export CFG_SCALE=1.0
export SELF_COND_CFG_SCALE=1.0
export COND_DROPOUT_PROB=0.0
export DENOISER_LOSS_WEIGHT=0.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_PROB=1.0
export DECODER_P_MEAN=$decoder_p_mean
export DECODER_P_STD=0.8
export DECODER_NOISE_SCALE=2.5
export DECODER_CONTENT_TOKEN_WEIGHT=$content_weight
export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.0
export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.0
export DECODER_CONDITION_PAIRING_MARGIN_WEIGHT=$pairing_weight
export DECODER_CONDITION_PAIRING_MARGIN=0.10

export GENERATION_T5_RETRIEVAL=1
export CHECKPOINT_METRIC=word_content_mean
export SAVE_TRAINABLE_ONLY_CHECKPOINT=1
export SAVE_FINAL_CHECKPOINT=1
export SAVE_EVAL_CHECKPOINTS=1
export EVAL_CHECKPOINT_TOP_K=2
export EVAL_CHECKPOINT_EVERY=0
export SEED=49

export RUN_TAG=meg_elfm_puresem_${source_label}_xattn_${arm}_ep${EPOCHS}_seed49_20260912
export OUTPUT_DIR=$runs_root/$RUN_TAG
export TARGET_LATENTS_CACHE=$interface/qc4wyals_predicted_train_valtail_t5small_latents_fp16.npy
export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1
export USE_WANDB=${USE_WANDB_OVERRIDE:-1}
export WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
export WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
export WANDB_RUN_GROUP=meg_elfm_pure_semantic_xattn_20260912
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_SAMPLE_EXAMPLES=20
export WANDB_NOTES="Pure qc4wyals-predicted-ADA ELF-M diffusion. Condition source=${source_label}. Frozen semantic winner and unchanged noise-to-latent flow; train only zero-init decoder-side cross-attention in final ${cross_blocks} blocks. No D'Ascoli or generated T5 source. Train2652/val110; test26 absent."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
