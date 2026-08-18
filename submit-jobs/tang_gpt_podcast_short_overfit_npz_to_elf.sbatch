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
export EVAL_EVERY=${EVAL_EVERY:-500}
export EVAL_NUM_EXAMPLES=${EVAL_NUM_EXAMPLES:-128}
export RETRIEVAL_EVAL_EVERY=${RETRIEVAL_EVAL_EVERY:-1000}
export RETRIEVAL_NUM_EXAMPLES=${RETRIEVAL_NUM_EXAMPLES:-0}
export RETRIEVAL_BATCH_SIZE=${RETRIEVAL_BATCH_SIZE:-256}
export NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-32}
export TRAIN_TIMESTEP_STEPS=${TRAIN_TIMESTEP_STEPS:-32}
export CONTEXT_LENGTH=${CONTEXT_LENGTH:-64}
export SEMANTIC_HIDDEN_DIM=${SEMANTIC_HIDDEN_DIM:-4096}
export LAST_N_BLOCKS=${LAST_N_BLOCKS:--1}
export LR=${LR:-1e-4}
export SEED=${SEED:-7}
export WANDB_SAMPLE_EXAMPLES=${WANDB_SAMPLE_EXAMPLES:-16}

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR" "$(dirname "$TARGET_LATENTS_CACHE")"

if ! command -v conda >/dev/null 2>&1; then
  for conda_sh in \
    "${CONDA_SH:-}" \
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
echo "TARGET_LATENTS_CACHE=$TARGET_LATENTS_CACHE"
echo "REEMBED_MODEL_NAME=$REEMBED_MODEL_NAME"

bash ./scripts/arc_run_npz_semantic_to_elf.sh \
  --device cuda \
  --input-key "$INPUT_KEY" \
  --sentence-key "$SENTENCE_KEY" \
  --num-examples 0 \
  --val-num-examples 0 \
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
  --last_n_blocks "$LAST_N_BLOCKS" \
  --lr "$LR" \
  --cfg_scale 1.0 \
  --self_cond_cfg_scale 1.0 \
  --cond_dropout_prob 0.0 \
  --denoiser_loss_weight 1.0 \
  --decoder_loss_weight 1.0 \
  --target-latents-mode cache \
  --target-latents-cache "$TARGET_LATENTS_CACHE" \
  --target-latents-cache-dtype float16 \
  --target-latent-cache-chunk-size 4096 \
  --target-latent-encode-batch-size 128 \
  --train-target-mask-mode full \
  --generation-t5-retrieval \
  --wandb_sample_examples "$WANDB_SAMPLE_EXAMPLES" \
  --save-best-checkpoint \
  --seed "$SEED" \
  --wandb_notes "Podcast overfit diagnostic. Train ELF from semantic embeddings to podcast text on 8-40 word Tang-GPT segment rows. Full ELF fine-tuning, K=64 context, full-mask target loss. If REEMBED_MODEL_NAME is set, the same rows are first re-embedded with that model for a matched embedding baseline."
