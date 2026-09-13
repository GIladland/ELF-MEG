#!/usr/bin/env bash
#SBATCH --job-name=tang_pod_short
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf-%j.out
#SBATCH --error=submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
export TMPDIR=${TMPDIR:-$DATA_ROOT/tmp}
export SKIP_ARC_SETUP=${SKIP_ARC_SETUP:-1}

export SOURCE_NPZ=${SOURCE_NPZ:-/data/engs-pnpl/glandau/elf-cache/tang_gpt/tang_gpt_seg10_text_pairs_train_words8to40_tanggpt.npz}
export NPZ_PATH=${NPZ_PATH:-$SOURCE_NPZ}
export INPUT_KEY=${INPUT_KEY:-input_embeddings}
export SENTENCE_KEY=${SENTENCE_KEY:-sentence}
export BRAIN_INPUT_NPZ=${BRAIN_INPUT_NPZ:-}
export BRAIN_TRAIN_KEY=${BRAIN_TRAIN_KEY:-train_x}
export BRAIN_VAL_KEY=${BRAIN_VAL_KEY:-val_x}
export BRAIN_VAL_ROLL=${BRAIN_VAL_ROLL:-0}
export SEMANTIC_VAL_ROLL=${SEMANTIC_VAL_ROLL:-0}
export BRAIN_MODEL_CHECKPOINT=${BRAIN_MODEL_CHECKPOINT:-}
export BRAIN_SEMANTIC_OUTPUT_DIM=${BRAIN_SEMANTIC_OUTPUT_DIM:-1536}
export BRAIN_PROJECTOR_MISMATCH=${BRAIN_PROJECTOR_MISMATCH:-error}
export BRAIN_HIDDEN_DIM=${BRAIN_HIDDEN_DIM:-2048}
export BRAIN_RES_BLOCKS=${BRAIN_RES_BLOCKS:-4}
export BRAIN_DROPOUT=${BRAIN_DROPOUT:-0.1}
export BRAIN_UNFREEZE=${BRAIN_UNFREEZE:-frozen}
export BRAIN_LR=${BRAIN_LR:-1e-5}
export BRAIN_LEXICAL_HEAD_CHECKPOINT=${BRAIN_LEXICAL_HEAD_CHECKPOINT:-}
export SEMANTIC_LEXICAL_HEAD_CHECKPOINT=${SEMANTIC_LEXICAL_HEAD_CHECKPOINT:-}
export SEMANTIC_ORDERED_HEAD_CHECKPOINT=${SEMANTIC_ORDERED_HEAD_CHECKPOINT:-}
export BRAIN_LEXICAL_TOPK=${BRAIN_LEXICAL_TOPK:-5}
export BRAIN_LEXICAL_TEMPERATURE=${BRAIN_LEXICAL_TEMPERATURE:-0.5}
export BRAIN_LEXICAL_CONTEXT_MODE=${BRAIN_LEXICAL_CONTEXT_MODE:-mixture}
export BRAIN_LEXICAL_PRIOR_SUBTRACTION=${BRAIN_LEXICAL_PRIOR_SUBTRACTION:-0.0}
export BRAIN_LEXICAL_DECODE_BIAS_STRENGTH=${BRAIN_LEXICAL_DECODE_BIAS_STRENGTH:-0.0}
export BRAIN_LEXICAL_MAX_PRIOR_PROBABILITY=${BRAIN_LEXICAL_MAX_PRIOR_PROBABILITY:-1.0}
export BRAIN_LEXICAL_DECODE_BIAS_ONCE=${BRAIN_LEXICAL_DECODE_BIAS_ONCE:-0}
export BRAIN_LEXICAL_DECODE_BIAS_MODE=${BRAIN_LEXICAL_DECODE_BIAS_MODE:-sequence}
export BRAIN_LEXICAL_DECODE_BIAS_MAX_POSITIONS=${BRAIN_LEXICAL_DECODE_BIAS_MAX_POSITIONS:-0}
export BRAIN_ORDERED_HEAD_CHECKPOINT=${BRAIN_ORDERED_HEAD_CHECKPOINT:-}
export BRAIN_ORDERED_PRIOR_SUBTRACTION=${BRAIN_ORDERED_PRIOR_SUBTRACTION:-0.0}
export BRAIN_ORDERED_MIN_PRIOR_PROBABILITY=${BRAIN_ORDERED_MIN_PRIOR_PROBABILITY:-0.0}
export BRAIN_ORDERED_MIN_MARGIN=${BRAIN_ORDERED_MIN_MARGIN:-0.0}
export BRAIN_ORDERED_DECODE_BIAS_STRENGTH=${BRAIN_ORDERED_DECODE_BIAS_STRENGTH:-0.0}
export BRAIN_ORDERED_DECODE_MAX_POSITIONS=${BRAIN_ORDERED_DECODE_MAX_POSITIONS:-0}
export MODEL=${MODEL:-ELF-B}
export CHECKPOINT_PATH=${CHECKPOINT_PATH:-embedded-language-flows/ELF-B-owt-torch}

export RUN_TAG=${RUN_TAG:-tang_gpt_seg10_podcast_words8to40_elf_fullmask_k64_full_1000ep}
export OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/$RUN_TAG}
export TARGET_LATENTS_CACHE=${TARGET_LATENTS_CACHE:-/data/engs-pnpl/glandau/elf-cache/tang_gpt/${RUN_TAG}_t5small_latents_fp16.npy}

export WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
export WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-tang_gpt_podcast_elf_overfit_short}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$RUN_TAG}
export WANDB_DIR=${WANDB_DIR:-$DATA_ROOT/wandb}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-$DATA_ROOT/wandb-data}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$DATA_ROOT/wandb-cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$DATA_ROOT/triton-cache}

export REEMBED_MODEL_NAME=${REEMBED_MODEL_NAME:-}
export REEMBED_POOLING=${REEMBED_POOLING:-mean}
export REEMBED_MAX_LENGTH=${REEMBED_MAX_LENGTH:-128}
export REEMBED_BATCH_SIZE=${REEMBED_BATCH_SIZE:-128}
export REEMBED_LOCAL_FILES_ONLY=${REEMBED_LOCAL_FILES_ONLY:-0}

export EPOCHS=${EPOCHS:-1000}
export BATCH_SIZE=${BATCH_SIZE:-16}
export NUM_EXAMPLES=${NUM_EXAMPLES:-0}
export VAL_NUM_EXAMPLES=${VAL_NUM_EXAMPLES:-0}
export EVAL_TARGET_LENGTH=${EVAL_TARGET_LENGTH:-0}
export CONDITION_SOURCE_KEY=${CONDITION_SOURCE_KEY:-condition_source}
export TRAIN_CONDITION_SOURCES=${TRAIN_CONDITION_SOURCES:-}
export TRAIN_BALANCE_KEY=${TRAIN_BALANCE_KEY:-}
export EVAL_EVERY=${EVAL_EVERY:-500}
export EVAL_NUM_EXAMPLES=${EVAL_NUM_EXAMPLES:-128}
export RETRIEVAL_EVAL_EVERY=${RETRIEVAL_EVAL_EVERY:-1000}
export RETRIEVAL_NUM_EXAMPLES=${RETRIEVAL_NUM_EXAMPLES:-0}
export GENERATION_T5_RETRIEVAL=${GENERATION_T5_RETRIEVAL:-1}
export RETRIEVAL_BATCH_SIZE=${RETRIEVAL_BATCH_SIZE:-256}
export NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-32}
export TRAIN_TIMESTEP_STEPS=${TRAIN_TIMESTEP_STEPS:-32}
export CONTEXT_LENGTH=${CONTEXT_LENGTH:-64}
export SEMANTIC_HIDDEN_DIM=${SEMANTIC_HIDDEN_DIM:-4096}
export SEMANTIC_ADAPTER_KIND=${SEMANTIC_ADAPTER_KIND:-flat}
export SEMANTIC_DELAY_TOKENS=${SEMANTIC_DELAY_TOKENS:-4}
export SEMANTIC_DELAY_LAYERS=${SEMANTIC_DELAY_LAYERS:-2}
export SEMANTIC_DELAY_HEADS=${SEMANTIC_DELAY_HEADS:-8}
export SEMANTIC_PRESERVE_DELAY_CONTEXT=${SEMANTIC_PRESERVE_DELAY_CONTEXT:-0}
export SEMANTIC_MAPPER_HIDDEN_DIM=${SEMANTIC_MAPPER_HIDDEN_DIM:-512}
export SEMANTIC_INPUT_PROJECTION_DIM=${SEMANTIC_INPUT_PROJECTION_DIM:-0}
export ADAPTER_DROPOUT=${ADAPTER_DROPOUT:-0.0}
export NORMALIZE_SEMANTIC_PROJECTION=${NORMALIZE_SEMANTIC_PROJECTION:-0}
export TRAIN_SEMANTIC_INPUT_ONLY=${TRAIN_SEMANTIC_INPUT_ONLY:-0}
export FREEZE_SEMANTIC_ADAPTER=${FREEZE_SEMANTIC_ADAPTER:-0}
export TRAIN_LEXICAL_CONTEXT_GATE_ONLY=${TRAIN_LEXICAL_CONTEXT_GATE_ONLY:-0}
export LAST_N_BLOCKS=${LAST_N_BLOCKS:--1}
export LR=${LR:-1e-4}
export ELF_LR=${ELF_LR:-0}
export ELF_LORA_RANK=${ELF_LORA_RANK:-0}
export ELF_LORA_ALPHA=${ELF_LORA_ALPHA:-16}
export ELF_LORA_DROPOUT=${ELF_LORA_DROPOUT:-0.05}
export ELF_LORA_TARGETS=${ELF_LORA_TARGETS:-attention}
export ELF_LORA_LAST_N_BLOCKS=${ELF_LORA_LAST_N_BLOCKS:--1}
export ELF_BRAIN_CROSS_ATTENTION_LAST_N_BLOCKS=${ELF_BRAIN_CROSS_ATTENTION_LAST_N_BLOCKS:-0}
export ELF_BRAIN_CROSS_ATTENTION_HEADS=${ELF_BRAIN_CROSS_ATTENTION_HEADS:-0}
export ELF_BRAIN_CROSS_ATTENTION_DROPOUT=${ELF_BRAIN_CROSS_ATTENTION_DROPOUT:-0.0}
export ELF_BRAIN_CROSS_ATTENTION_ONLY=${ELF_BRAIN_CROSS_ATTENTION_ONLY:-0}
export ELF_BRAIN_CROSS_ATTENTION_DECODER_ONLY=${ELF_BRAIN_CROSS_ATTENTION_DECODER_ONLY:-0}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
export LR_SCHEDULE=${LR_SCHEDULE:-constant}
export WARMUP_STEPS=${WARMUP_STEPS:-0}
export MIN_LR_RATIO=${MIN_LR_RATIO:-0.01}
export CFG_SCALE=${CFG_SCALE:-1.0}
export SELF_COND_CFG_SCALE=${SELF_COND_CFG_SCALE:-1.0}
export COND_DROPOUT_PROB=${COND_DROPOUT_PROB:-0.0}
export DENOISER_LOSS_WEIGHT=${DENOISER_LOSS_WEIGHT:-1.0}
export DECODER_LOSS_WEIGHT=${DECODER_LOSS_WEIGHT:-1.0}
export DECODER_CONTENT_TOKEN_WEIGHT=${DECODER_CONTENT_TOKEN_WEIGHT:-1.0}
export DECODER_CONTENT_BOW_LOSS_WEIGHT=${DECODER_CONTENT_BOW_LOSS_WEIGHT:-0.0}
export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=${DECODER_CONTENT_PRECISION_LOSS_WEIGHT:-0.0}
export DECODER_CONTENT_PRECISION_TOPK=${DECODER_CONTENT_PRECISION_TOPK:-32}
export DECODER_CONDITION_PAIRING_MARGIN_WEIGHT=${DECODER_CONDITION_PAIRING_MARGIN_WEIGHT:-0.0}
export DECODER_CONDITION_PAIRING_MARGIN=${DECODER_CONDITION_PAIRING_MARGIN:-0.1}
export DECODER_PROB=${DECODER_PROB:-0.5}
export DECODER_P_MEAN=${DECODER_P_MEAN:-0.8}
export DECODER_P_STD=${DECODER_P_STD:-0.8}
export DECODER_NOISE_SCALE=${DECODER_NOISE_SCALE:-2.5}
export DECODER_POSITION_PRIOR_STRENGTH=${DECODER_POSITION_PRIOR_STRENGTH:-0.0}
export DECODER_POSITION_PRIOR_SMOOTHING=${DECODER_POSITION_PRIOR_SMOOTHING:-0.25}
export DECODER_POSITION_PRIOR_CLIP=${DECODER_POSITION_PRIOR_CLIP:-5.0}
export DECODER_FLOW_LATENT_CACHE=${DECODER_FLOW_LATENT_CACHE:-}
export DECODER_FLOW_LATENT_BATCH_SIZE=${DECODER_FLOW_LATENT_BATCH_SIZE:-16}
export DECODER_FLOW_LATENT_SEED=${DECODER_FLOW_LATENT_SEED:-66}
export DECODER_FLOW_LATENT_DIRECT=${DECODER_FLOW_LATENT_DIRECT:-0}
export SEMANTIC_ALIGNMENT_TARGET_NPZ=${SEMANTIC_ALIGNMENT_TARGET_NPZ:-}
export SEMANTIC_ALIGNMENT_TARGET_KEY=${SEMANTIC_ALIGNMENT_TARGET_KEY:-input_embeddings}
export ALLOW_SEMANTIC_ALIGNMENT_SENTENCE_MISMATCH=${ALLOW_SEMANTIC_ALIGNMENT_SENTENCE_MISMATCH:-0}
export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=${SEMANTIC_ALIGNMENT_LOSS_WEIGHT:-0.0}
export SEMANTIC_ALIGNMENT_LOSS_TYPE=${SEMANTIC_ALIGNMENT_LOSS_TYPE:-cosine}
export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=${SEMANTIC_CONTRASTIVE_LOSS_WEIGHT:-0.0}
export SEMANTIC_CONTRASTIVE_TEMPERATURE=${SEMANTIC_CONTRASTIVE_TEMPERATURE:-0.07}
export SEMANTIC_CONTEXT_LOSS_WEIGHT=${SEMANTIC_CONTEXT_LOSS_WEIGHT:-0.0}
export SEMANTIC_CONTENT_LOSS_WEIGHT=${SEMANTIC_CONTENT_LOSS_WEIGHT:-0.0}
export SEMANTIC_CONTENT_MIN_FREQUENCY=${SEMANTIC_CONTENT_MIN_FREQUENCY:-5}
export SEMANTIC_CONTENT_MAX_VOCABULARY=${SEMANTIC_CONTENT_MAX_VOCABULARY:-3000}
export SEMANTIC_CONTENT_NEGATIVE_TOPK=${SEMANTIC_CONTENT_NEGATIVE_TOPK:-32}
export SEMANTIC_CONTENT_TEMPERATURE=${SEMANTIC_CONTENT_TEMPERATURE:-0.1}
export SEMANTIC_RAW_ANCHOR_LOSS_WEIGHT=${SEMANTIC_RAW_ANCHOR_LOSS_WEIGHT:-0.0}
export SEMANTIC_GEOMETRY_LOSS_WEIGHT=${SEMANTIC_GEOMETRY_LOSS_WEIGHT:-0.0}
export SEMANTIC_RANK_DISTILL_LOSS_WEIGHT=${SEMANTIC_RANK_DISTILL_LOSS_WEIGHT:-0.0}
export SEMANTIC_RANK_DISTILL_TEMPERATURE=${SEMANTIC_RANK_DISTILL_TEMPERATURE:-0.07}
export SEMANTIC_OBJECTIVE_ONLY=${SEMANTIC_OBJECTIVE_ONLY:-0}
export CURRICULUM_SEMANTIC_ONLY_EPOCHS=${CURRICULUM_SEMANTIC_ONLY_EPOCHS:-0}
export CURRICULUM_DENOISER_LOSS_WEIGHT=${CURRICULUM_DENOISER_LOSS_WEIGHT:-0.1}
export CURRICULUM_DECODER_LOSS_WEIGHT=${CURRICULUM_DECODER_LOSS_WEIGHT:-0.1}
export TRAIN_TARGET_MASK_MODE=${TRAIN_TARGET_MASK_MODE:-full}
export FREEZE_ELF=${FREEZE_ELF:-0}
export EVAL_ONLY=${EVAL_ONLY:-0}
export USE_WANDB=${USE_WANDB:-1}
export SEED=${SEED:-7}
export WANDB_SAMPLE_EXAMPLES=${WANDB_SAMPLE_EXAMPLES:-16}
export CHECKPOINT_METRIC=${CHECKPOINT_METRIC:-generation_t5_top1}
export INIT_E2E_CHECKPOINT=${INIT_E2E_CHECKPOINT:-}
export INIT_E2E_ADAPTER_MISMATCH=${INIT_E2E_ADAPTER_MISMATCH:-error}
export SAVE_TRAINABLE_ONLY_CHECKPOINT=${SAVE_TRAINABLE_ONLY_CHECKPOINT:-0}
export SAVE_FINAL_CHECKPOINT=${SAVE_FINAL_CHECKPOINT:-0}
export SAVE_EVAL_CHECKPOINTS=${SAVE_EVAL_CHECKPOINTS:-0}
export EVAL_CHECKPOINT_TOP_K=${EVAL_CHECKPOINT_TOP_K:-1}
export EVAL_CHECKPOINT_EVERY=${EVAL_CHECKPOINT_EVERY:-0}
export WANDB_NOTES=${WANDB_NOTES:-Podcast overfit diagnostic. Train ELF from semantic embeddings to podcast text on short Tang-GPT story chunks. Full ELF fine-tuning, K=64 context, full-mask target loss. If REEMBED_MODEL_NAME is set, the same rows are first re-embedded with that model for a matched embedding baseline.}

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR" "$(dirname "$TARGET_LATENTS_CACHE")"

if ! command -v conda >/dev/null 2>&1; then
  for conda_sh in \
    "${CONDA_SH:-}" \
    "/apps/system/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/data/engs-pnpl/glandau/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -n "$conda_sh" && -f "$conda_sh" ]]; then
      source "$conda_sh"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found; set CONDA_SH before launching." >&2
  exit 127
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_ROOT/xdg-cache}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$DATA_ROOT/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/pip-cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$DATA_ROOT/conda-pkgs}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

cd "$PROJECT_ROOT"
ENV_STAMP_PATH="$TMP_ENV_ROOT/.elf_env_stamp"
if [[ "$SKIP_ARC_SETUP" == "1" && -d "$TMP_ENV_ROOT/bin" && -f "$ENV_STAMP_PATH" && "$(cat "$ENV_STAMP_PATH")" == "$ENV_STAMP" ]]; then
  echo "Skipping ARC env setup; reusing TMP_ENV_ROOT=$TMP_ENV_ROOT ENV_STAMP=$ENV_STAMP"
else
  PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" TMP_ENV_ROOT="$TMP_ENV_ROOT" ENV_STAMP="$ENV_STAMP" \
    bash "$PROJECT_ROOT/scripts/arc_setup_elf_env.sh"
fi
conda activate "$TMP_ENV_ROOT"

if [[ -n "$REEMBED_MODEL_NAME" && ! -f "$NPZ_PATH" ]]; then
  reembed_args=(
    scripts/reembed_semantic_npz_sentences.py
    --input-npz "$SOURCE_NPZ"
    --output "$NPZ_PATH"
    --input-key "$INPUT_KEY"
    --sentence-key "$SENTENCE_KEY"
    --embedding-model-name "$REEMBED_MODEL_NAME"
    --pooling "$REEMBED_POOLING"
    --max-length "$REEMBED_MAX_LENGTH"
    --batch-size "$REEMBED_BATCH_SIZE"
    --device cuda
  )
  if [[ "$REEMBED_LOCAL_FILES_ONLY" == "1" ]]; then
    reembed_args+=(--local-files-only)
  fi
  python "${reembed_args[@]}"
fi

echo "RUN_TAG=$RUN_TAG"
echo "SOURCE_NPZ=$SOURCE_NPZ"
echo "NPZ_PATH=$NPZ_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "BRAIN_INPUT_NPZ=$BRAIN_INPUT_NPZ BRAIN_MODEL_CHECKPOINT=$BRAIN_MODEL_CHECKPOINT BRAIN_UNFREEZE=$BRAIN_UNFREEZE BRAIN_LR=$BRAIN_LR"
echo "SEMANTIC_LEXICAL_HEAD_CHECKPOINT=$SEMANTIC_LEXICAL_HEAD_CHECKPOINT SEMANTIC_ORDERED_HEAD_CHECKPOINT=$SEMANTIC_ORDERED_HEAD_CHECKPOINT BRAIN_LEXICAL_TOPK=$BRAIN_LEXICAL_TOPK BRAIN_LEXICAL_PRIOR_SUBTRACTION=$BRAIN_LEXICAL_PRIOR_SUBTRACTION BRAIN_LEXICAL_DECODE_BIAS_STRENGTH=$BRAIN_LEXICAL_DECODE_BIAS_STRENGTH"
echo "MODEL=$MODEL CHECKPOINT_PATH=$CHECKPOINT_PATH"
echo "TARGET_LATENTS_CACHE=$TARGET_LATENTS_CACHE"
echo "REEMBED_MODEL_NAME=$REEMBED_MODEL_NAME"
echo "INIT_E2E_CHECKPOINT=$INIT_E2E_CHECKPOINT"
echo "EVAL_ONLY=$EVAL_ONLY USE_WANDB=$USE_WANDB"
echo "CFG_SCALE=$CFG_SCALE SELF_COND_CFG_SCALE=$SELF_COND_CFG_SCALE"
echo "DENOISER_LOSS_WEIGHT=$DENOISER_LOSS_WEIGHT DECODER_LOSS_WEIGHT=$DECODER_LOSS_WEIGHT DECODER_PROB=$DECODER_PROB DECODER_P_MEAN=$DECODER_P_MEAN DECODER_P_STD=$DECODER_P_STD DECODER_NOISE_SCALE=$DECODER_NOISE_SCALE"
echo "DECODER_POSITION_PRIOR_STRENGTH=$DECODER_POSITION_PRIOR_STRENGTH DECODER_POSITION_PRIOR_SMOOTHING=$DECODER_POSITION_PRIOR_SMOOTHING DECODER_POSITION_PRIOR_CLIP=$DECODER_POSITION_PRIOR_CLIP"
echo "DECODER_CONTENT_TOKEN_WEIGHT=$DECODER_CONTENT_TOKEN_WEIGHT"
echo "DECODER_CONTENT_BOW_LOSS_WEIGHT=$DECODER_CONTENT_BOW_LOSS_WEIGHT"
echo "DECODER_CONTENT_PRECISION_LOSS_WEIGHT=$DECODER_CONTENT_PRECISION_LOSS_WEIGHT DECODER_CONTENT_PRECISION_TOPK=$DECODER_CONTENT_PRECISION_TOPK"
echo "DECODER_CONDITION_PAIRING_MARGIN_WEIGHT=$DECODER_CONDITION_PAIRING_MARGIN_WEIGHT DECODER_CONDITION_PAIRING_MARGIN=$DECODER_CONDITION_PAIRING_MARGIN"
echo "SEMANTIC_ALIGNMENT_TARGET_NPZ=$SEMANTIC_ALIGNMENT_TARGET_NPZ ALIGNMENT_WEIGHT=$SEMANTIC_ALIGNMENT_LOSS_WEIGHT CONTRASTIVE_WEIGHT=$SEMANTIC_CONTRASTIVE_LOSS_WEIGHT CONTEXT_WEIGHT=$SEMANTIC_CONTEXT_LOSS_WEIGHT"
echo "SEMANTIC_RAW_ANCHOR_WEIGHT=$SEMANTIC_RAW_ANCHOR_LOSS_WEIGHT SEMANTIC_GEOMETRY_WEIGHT=$SEMANTIC_GEOMETRY_LOSS_WEIGHT SEMANTIC_RANK_DISTILL_WEIGHT=$SEMANTIC_RANK_DISTILL_LOSS_WEIGHT"
echo "CURRICULUM_SEMANTIC_ONLY_EPOCHS=$CURRICULUM_SEMANTIC_ONLY_EPOCHS CURRICULUM_DENOISER_WEIGHT=$CURRICULUM_DENOISER_LOSS_WEIGHT CURRICULUM_DECODER_WEIGHT=$CURRICULUM_DECODER_LOSS_WEIGHT"
echo "SEMANTIC_ADAPTER_KIND=$SEMANTIC_ADAPTER_KIND SEMANTIC_DELAY_TOKENS=$SEMANTIC_DELAY_TOKENS SEMANTIC_DELAY_LAYERS=$SEMANTIC_DELAY_LAYERS SEMANTIC_DELAY_HEADS=$SEMANTIC_DELAY_HEADS"
echo "SEMANTIC_MAPPER_HIDDEN_DIM=$SEMANTIC_MAPPER_HIDDEN_DIM"
echo "TRAIN_SEMANTIC_INPUT_ONLY=$TRAIN_SEMANTIC_INPUT_ONLY FREEZE_SEMANTIC_ADAPTER=$FREEZE_SEMANTIC_ADAPTER NORMALIZE_SEMANTIC_PROJECTION=$NORMALIZE_SEMANTIC_PROJECTION"
echo "ELF_LR=$ELF_LR ELF_LORA_RANK=$ELF_LORA_RANK ELF_LORA_ALPHA=$ELF_LORA_ALPHA ELF_LORA_DROPOUT=$ELF_LORA_DROPOUT ELF_LORA_TARGETS=$ELF_LORA_TARGETS ELF_LORA_LAST_N_BLOCKS=$ELF_LORA_LAST_N_BLOCKS"
echo "ELF_BRAIN_CROSS_ATTENTION_LAST_N_BLOCKS=$ELF_BRAIN_CROSS_ATTENTION_LAST_N_BLOCKS ELF_BRAIN_CROSS_ATTENTION_HEADS=$ELF_BRAIN_CROSS_ATTENTION_HEADS ELF_BRAIN_CROSS_ATTENTION_DROPOUT=$ELF_BRAIN_CROSS_ATTENTION_DROPOUT ELF_BRAIN_CROSS_ATTENTION_DECODER_ONLY=$ELF_BRAIN_CROSS_ATTENTION_DECODER_ONLY"
echo "TRAIN_CONDITION_SOURCES=$TRAIN_CONDITION_SOURCES CONDITION_SOURCE_KEY=$CONDITION_SOURCE_KEY"
echo "TRAIN_BALANCE_KEY=$TRAIN_BALANCE_KEY"

init_args=()
if [[ -n "$INIT_E2E_CHECKPOINT" ]]; then
  init_args+=(--init-e2e-checkpoint "$INIT_E2E_CHECKPOINT")
  init_args+=(--init-e2e-adapter-mismatch "$INIT_E2E_ADAPTER_MISMATCH")
fi
eval_args=()
if [[ "$EVAL_ONLY" == "1" ]]; then
  eval_args+=(--eval-only)
fi
freeze_args=()
if [[ "$FREEZE_ELF" == "1" ]]; then
  freeze_args+=(--freeze-elf)
fi
brain_cross_attention_args=()
if [[ "$ELF_BRAIN_CROSS_ATTENTION_ONLY" == "1" ]]; then
  brain_cross_attention_args+=(--elf-brain-cross-attention-only)
fi
decoder_flow_latent_args=()
if [[ -n "$DECODER_FLOW_LATENT_CACHE" ]]; then
  decoder_flow_latent_args+=(--decoder-flow-latent-cache "$DECODER_FLOW_LATENT_CACHE")
  decoder_flow_latent_args+=(--decoder-flow-latent-batch-size "$DECODER_FLOW_LATENT_BATCH_SIZE")
  decoder_flow_latent_args+=(--decoder-flow-latent-seed "$DECODER_FLOW_LATENT_SEED")
fi
if [[ "$DECODER_FLOW_LATENT_DIRECT" == "1" ]]; then
  decoder_flow_latent_args+=(--decoder-flow-latent-direct)
fi
if [[ "$ELF_BRAIN_CROSS_ATTENTION_DECODER_ONLY" == "1" ]]; then
  brain_cross_attention_args+=(--elf-brain-cross-attention-decoder-only)
fi
wandb_args=()
if [[ "$USE_WANDB" == "1" ]]; then
  wandb_args+=(--use_wandb)
fi
checkpoint_args=(--save-best-checkpoint)
if [[ "$SAVE_TRAINABLE_ONLY_CHECKPOINT" == "1" ]]; then
  checkpoint_args+=(--save-trainable-only-checkpoint)
fi
if [[ "$SAVE_FINAL_CHECKPOINT" == "1" ]]; then
  checkpoint_args+=(--save-final-checkpoint)
fi
if [[ "$SAVE_EVAL_CHECKPOINTS" == "1" ]]; then
  checkpoint_args+=(--save-eval-checkpoints)
  checkpoint_args+=(--eval-checkpoint-top-k "$EVAL_CHECKPOINT_TOP_K")
  checkpoint_args+=(--eval-checkpoint-every "$EVAL_CHECKPOINT_EVERY")
fi
semantic_train_args=()
if [[ "$TRAIN_SEMANTIC_INPUT_ONLY" == "1" ]]; then
  semantic_train_args+=(--train-semantic-input-only)
fi
if [[ "$FREEZE_SEMANTIC_ADAPTER" == "1" ]]; then
  semantic_train_args+=(--freeze-semantic-adapter)
fi
if [[ "$TRAIN_LEXICAL_CONTEXT_GATE_ONLY" == "1" ]]; then
  semantic_train_args+=(--train-lexical-context-gate-only)
fi
if [[ "$NORMALIZE_SEMANTIC_PROJECTION" == "1" ]]; then
  semantic_train_args+=(--normalize-semantic-projection)
fi
if [[ "$SEMANTIC_PRESERVE_DELAY_CONTEXT" == "1" ]]; then
  semantic_train_args+=(--semantic-preserve-delay-context)
fi
semantic_alignment_args=()
semantic_alignment_args+=(--semantic-raw-anchor-loss-weight "$SEMANTIC_RAW_ANCHOR_LOSS_WEIGHT")
semantic_alignment_args+=(--semantic-geometry-loss-weight "$SEMANTIC_GEOMETRY_LOSS_WEIGHT")
semantic_alignment_args+=(--semantic-rank-distill-loss-weight "$SEMANTIC_RANK_DISTILL_LOSS_WEIGHT")
semantic_alignment_args+=(--semantic-rank-distill-temperature "$SEMANTIC_RANK_DISTILL_TEMPERATURE")
if [[ -n "$SEMANTIC_ALIGNMENT_TARGET_NPZ" ]]; then
  semantic_alignment_args+=(--semantic-alignment-target-npz "$SEMANTIC_ALIGNMENT_TARGET_NPZ")
  semantic_alignment_args+=(--semantic-alignment-target-key "$SEMANTIC_ALIGNMENT_TARGET_KEY")
  semantic_alignment_args+=(--semantic-alignment-loss-weight "$SEMANTIC_ALIGNMENT_LOSS_WEIGHT")
  semantic_alignment_args+=(--semantic-alignment-loss-type "$SEMANTIC_ALIGNMENT_LOSS_TYPE")
  semantic_alignment_args+=(--semantic-contrastive-loss-weight "$SEMANTIC_CONTRASTIVE_LOSS_WEIGHT")
  semantic_alignment_args+=(--semantic-contrastive-temperature "$SEMANTIC_CONTRASTIVE_TEMPERATURE")
  semantic_alignment_args+=(--semantic-context-loss-weight "$SEMANTIC_CONTEXT_LOSS_WEIGHT")
  semantic_alignment_args+=(--semantic-content-loss-weight "$SEMANTIC_CONTENT_LOSS_WEIGHT")
  semantic_alignment_args+=(--semantic-content-min-frequency "$SEMANTIC_CONTENT_MIN_FREQUENCY")
  semantic_alignment_args+=(--semantic-content-max-vocabulary "$SEMANTIC_CONTENT_MAX_VOCABULARY")
  semantic_alignment_args+=(--semantic-content-negative-topk "$SEMANTIC_CONTENT_NEGATIVE_TOPK")
  if [[ "$ALLOW_SEMANTIC_ALIGNMENT_SENTENCE_MISMATCH" == "1" ]]; then
    semantic_alignment_args+=(--allow-semantic-alignment-sentence-mismatch)
  fi
  semantic_alignment_args+=(--semantic-content-temperature "$SEMANTIC_CONTENT_TEMPERATURE")
  semantic_alignment_args+=(--curriculum-semantic-only-epochs "$CURRICULUM_SEMANTIC_ONLY_EPOCHS")
  semantic_alignment_args+=(--curriculum-denoiser-loss-weight "$CURRICULUM_DENOISER_LOSS_WEIGHT")
  semantic_alignment_args+=(--curriculum-decoder-loss-weight "$CURRICULUM_DECODER_LOSS_WEIGHT")
fi
if [[ "$SEMANTIC_OBJECTIVE_ONLY" == "1" ]]; then
  semantic_alignment_args+=(--semantic-objective-only)
fi
condition_source_args=()
if [[ -n "$TRAIN_CONDITION_SOURCES" ]]; then
  read -r -a requested_condition_sources <<< "$TRAIN_CONDITION_SOURCES"
  condition_source_args+=(--condition-source-key "$CONDITION_SOURCE_KEY")
  condition_source_args+=(--train-condition-sources "${requested_condition_sources[@]}")
fi
if [[ -n "$TRAIN_BALANCE_KEY" ]]; then
  condition_source_args+=(--train-balance-key "$TRAIN_BALANCE_KEY")
fi
brain_args=()
if [[ -n "$BRAIN_INPUT_NPZ" ]]; then
  brain_args+=(--brain-input-npz "$BRAIN_INPUT_NPZ")
  brain_args+=(--brain-train-key "$BRAIN_TRAIN_KEY")
  brain_args+=(--brain-val-key "$BRAIN_VAL_KEY")
  brain_args+=(--brain-val-roll "$BRAIN_VAL_ROLL")
  brain_args+=(--brain-model-checkpoint "$BRAIN_MODEL_CHECKPOINT")
  brain_args+=(--brain-semantic-output-dim "$BRAIN_SEMANTIC_OUTPUT_DIM")
  brain_args+=(--brain-projector-mismatch "$BRAIN_PROJECTOR_MISMATCH")
  brain_args+=(--brain-hidden-dim "$BRAIN_HIDDEN_DIM")
  brain_args+=(--brain-res-blocks "$BRAIN_RES_BLOCKS")
  brain_args+=(--brain-dropout "$BRAIN_DROPOUT")
  brain_args+=(--brain-unfreeze "$BRAIN_UNFREEZE")
  brain_args+=(--brain-lr "$BRAIN_LR")
  if [[ -n "$BRAIN_LEXICAL_HEAD_CHECKPOINT" ]]; then
    brain_args+=(--brain-lexical-head-checkpoint "$BRAIN_LEXICAL_HEAD_CHECKPOINT")
    brain_args+=(--brain-lexical-topk "$BRAIN_LEXICAL_TOPK")
    brain_args+=(--brain-lexical-temperature "$BRAIN_LEXICAL_TEMPERATURE")
    brain_args+=(--brain-lexical-context-mode "$BRAIN_LEXICAL_CONTEXT_MODE")
    brain_args+=(--brain-lexical-prior-subtraction "$BRAIN_LEXICAL_PRIOR_SUBTRACTION")
    brain_args+=(--brain-lexical-decode-bias-strength "$BRAIN_LEXICAL_DECODE_BIAS_STRENGTH")
    brain_args+=(--brain-lexical-max-prior-probability "$BRAIN_LEXICAL_MAX_PRIOR_PROBABILITY")
    brain_args+=(--brain-lexical-decode-bias-mode "$BRAIN_LEXICAL_DECODE_BIAS_MODE")
    brain_args+=(--brain-lexical-decode-bias-max-positions "$BRAIN_LEXICAL_DECODE_BIAS_MAX_POSITIONS")
    if [[ -n "$BRAIN_ORDERED_HEAD_CHECKPOINT" ]]; then
      brain_args+=(--brain-ordered-head-checkpoint "$BRAIN_ORDERED_HEAD_CHECKPOINT")
      brain_args+=(--brain-ordered-prior-subtraction "$BRAIN_ORDERED_PRIOR_SUBTRACTION")
      brain_args+=(--brain-ordered-min-prior-probability "$BRAIN_ORDERED_MIN_PRIOR_PROBABILITY")
      brain_args+=(--brain-ordered-min-margin "$BRAIN_ORDERED_MIN_MARGIN")
      brain_args+=(--brain-ordered-decode-bias-strength "$BRAIN_ORDERED_DECODE_BIAS_STRENGTH")
      brain_args+=(--brain-ordered-decode-max-positions "$BRAIN_ORDERED_DECODE_MAX_POSITIONS")
    fi
    if [[ "$BRAIN_LEXICAL_DECODE_BIAS_ONCE" == "1" ]]; then
      brain_args+=(--brain-lexical-decode-bias-once)
    fi
  fi
fi
if [[ -n "$SEMANTIC_LEXICAL_HEAD_CHECKPOINT" ]]; then
  brain_args+=(--semantic-lexical-head-checkpoint "$SEMANTIC_LEXICAL_HEAD_CHECKPOINT")
  brain_args+=(--brain-lexical-topk "$BRAIN_LEXICAL_TOPK")
  brain_args+=(--brain-lexical-temperature "$BRAIN_LEXICAL_TEMPERATURE")
  brain_args+=(--brain-lexical-context-mode "$BRAIN_LEXICAL_CONTEXT_MODE")
  brain_args+=(--brain-lexical-prior-subtraction "$BRAIN_LEXICAL_PRIOR_SUBTRACTION")
  brain_args+=(--brain-lexical-decode-bias-strength "$BRAIN_LEXICAL_DECODE_BIAS_STRENGTH")
  brain_args+=(--brain-lexical-max-prior-probability "$BRAIN_LEXICAL_MAX_PRIOR_PROBABILITY")
  brain_args+=(--brain-lexical-decode-bias-mode "$BRAIN_LEXICAL_DECODE_BIAS_MODE")
  brain_args+=(--brain-lexical-decode-bias-max-positions "$BRAIN_LEXICAL_DECODE_BIAS_MAX_POSITIONS")
  if [[ "$BRAIN_LEXICAL_DECODE_BIAS_ONCE" == "1" ]]; then
    brain_args+=(--brain-lexical-decode-bias-once)
  fi
  if [[ -n "$SEMANTIC_ORDERED_HEAD_CHECKPOINT" ]]; then
    brain_args+=(--semantic-ordered-head-checkpoint "$SEMANTIC_ORDERED_HEAD_CHECKPOINT")
    brain_args+=(--brain-ordered-prior-subtraction "$BRAIN_ORDERED_PRIOR_SUBTRACTION")
    brain_args+=(--brain-ordered-min-prior-probability "$BRAIN_ORDERED_MIN_PRIOR_PROBABILITY")
    brain_args+=(--brain-ordered-min-margin "$BRAIN_ORDERED_MIN_MARGIN")
    brain_args+=(--brain-ordered-decode-bias-strength "$BRAIN_ORDERED_DECODE_BIAS_STRENGTH")
    brain_args+=(--brain-ordered-decode-max-positions "$BRAIN_ORDERED_DECODE_MAX_POSITIONS")
  fi
fi
generation_retrieval_args=()
if [[ "$GENERATION_T5_RETRIEVAL" == "1" ]]; then
  generation_retrieval_args+=(--generation-t5-retrieval)
fi

python scripts/train_npz_semantic_to_elf.py \
  --npz-path "$NPZ_PATH" \
  --output_dir "$OUTPUT_DIR" \
  "${wandb_args[@]}" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_group "$WANDB_RUN_GROUP" \
  --wandb_run_name "$WANDB_RUN_NAME" \
  --device cuda \
  --input-key "$INPUT_KEY" \
  --sentence-key "$SENTENCE_KEY" \
  --semantic-val-roll "$SEMANTIC_VAL_ROLL" \
  "${brain_args[@]}" \
  --model "$MODEL" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --num-examples "$NUM_EXAMPLES" \
  --val-num-examples "$VAL_NUM_EXAMPLES" \
  --eval-target-length "$EVAL_TARGET_LENGTH" \
  "${condition_source_args[@]}" \
  --eval-num-examples "$EVAL_NUM_EXAMPLES" \
  --retrieval-num-examples "$RETRIEVAL_NUM_EXAMPLES" \
  --retrieval-eval-every "$RETRIEVAL_EVAL_EVERY" \
  --retrieval-batch-size "$RETRIEVAL_BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --eval_every "$EVAL_EVERY" \
  --num_sampling_steps "$NUM_SAMPLING_STEPS" \
  --train-timestep-mode sampling_schedule \
  --train-timestep-steps "$TRAIN_TIMESTEP_STEPS" \
  --context_length "$CONTEXT_LENGTH" \
  --semantic-hidden-dim "$SEMANTIC_HIDDEN_DIM" \
  --semantic-adapter-kind "$SEMANTIC_ADAPTER_KIND" \
  --semantic-delay-tokens "$SEMANTIC_DELAY_TOKENS" \
  --semantic-delay-layers "$SEMANTIC_DELAY_LAYERS" \
  --semantic-delay-heads "$SEMANTIC_DELAY_HEADS" \
  --semantic-mapper-hidden-dim "$SEMANTIC_MAPPER_HIDDEN_DIM" \
  --semantic-input-projection-dim "$SEMANTIC_INPUT_PROJECTION_DIM" \
  --adapter-dropout "$ADAPTER_DROPOUT" \
  "${semantic_train_args[@]}" \
  "${semantic_alignment_args[@]}" \
  --last_n_blocks "$LAST_N_BLOCKS" \
  --lr "$LR" \
  --elf-lr "$ELF_LR" \
  --elf-lora-rank "$ELF_LORA_RANK" \
  --elf-lora-alpha "$ELF_LORA_ALPHA" \
  --elf-lora-dropout "$ELF_LORA_DROPOUT" \
  --elf-lora-targets "$ELF_LORA_TARGETS" \
  --elf-lora-last-n-blocks "$ELF_LORA_LAST_N_BLOCKS" \
  --elf-brain-cross-attention-last-n-blocks "$ELF_BRAIN_CROSS_ATTENTION_LAST_N_BLOCKS" \
  --elf-brain-cross-attention-heads "$ELF_BRAIN_CROSS_ATTENTION_HEADS" \
  --elf-brain-cross-attention-dropout "$ELF_BRAIN_CROSS_ATTENTION_DROPOUT" \
  "${brain_cross_attention_args[@]}" \
  --weight_decay "$WEIGHT_DECAY" \
  --lr-schedule "$LR_SCHEDULE" \
  --warmup-steps "$WARMUP_STEPS" \
  --min-lr-ratio "$MIN_LR_RATIO" \
  --cfg_scale "$CFG_SCALE" \
  --self_cond_cfg_scale "$SELF_COND_CFG_SCALE" \
  --cond_dropout_prob "$COND_DROPOUT_PROB" \
  --denoiser_loss_weight "$DENOISER_LOSS_WEIGHT" \
  --decoder_loss_weight "$DECODER_LOSS_WEIGHT" \
  --decoder-content-token-weight "$DECODER_CONTENT_TOKEN_WEIGHT" \
  --decoder-content-bow-loss-weight "$DECODER_CONTENT_BOW_LOSS_WEIGHT" \
  --decoder-content-precision-loss-weight "$DECODER_CONTENT_PRECISION_LOSS_WEIGHT" \
  --decoder-content-precision-topk "$DECODER_CONTENT_PRECISION_TOPK" \
  --decoder-condition-pairing-margin-weight "$DECODER_CONDITION_PAIRING_MARGIN_WEIGHT" \
  --decoder-condition-pairing-margin "$DECODER_CONDITION_PAIRING_MARGIN" \
  --decoder_prob "$DECODER_PROB" \
  --decoder_p_mean "$DECODER_P_MEAN" \
  --decoder_p_std "$DECODER_P_STD" \
  --decoder_noise_scale "$DECODER_NOISE_SCALE" \
  --decoder-position-prior-strength "$DECODER_POSITION_PRIOR_STRENGTH" \
  --decoder-position-prior-smoothing "$DECODER_POSITION_PRIOR_SMOOTHING" \
  --decoder-position-prior-clip "$DECODER_POSITION_PRIOR_CLIP" \
  "${decoder_flow_latent_args[@]}" \
  --target-latents-mode cache \
  --target-latents-cache "$TARGET_LATENTS_CACHE" \
  --target-latents-cache-dtype float16 \
  --target-latent-cache-chunk-size 4096 \
  --target-latent-encode-batch-size 128 \
  --train-target-mask-mode "$TRAIN_TARGET_MASK_MODE" \
  "${freeze_args[@]}" \
  "${init_args[@]}" \
  "${eval_args[@]}" \
  "${generation_retrieval_args[@]}" \
  --wandb_sample_examples "$WANDB_SAMPLE_EXAMPLES" \
  --checkpoint-metric "$CHECKPOINT_METRIC" \
  "${checkpoint_args[@]}" \
  --seed "$SEED" \
  --wandb_notes "$WANDB_NOTES"
