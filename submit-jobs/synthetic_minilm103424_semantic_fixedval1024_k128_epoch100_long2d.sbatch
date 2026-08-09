#!/usr/bin/env bash
#SBATCH --job-name=minilm100k_k128
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/synthetic_minilm103424_semantic_fixedval1024_k128_epoch100_long2d-%j.out
#SBATCH --error=submit-jobs/synthetic_minilm103424_semantic_fixedval1024_k128_epoch100_long2d-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
export BASE_SYNTHETIC_OUTPUT=/data/engs-pnpl/glandau/elf-cache/synthetic/synthetic_minilm_103424_seed7.npz
export FIXEDVAL_SYNTHETIC_OUTPUT=/data/engs-pnpl/glandau/elf-cache/synthetic/synthetic_minilm_103424_seed7_fixedval10240_1024.npz
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/synthetic_minilm103424_semantic_fixedval1024_k128_epoch100_long2d
export SYNTHETIC_COUNT=103424
export SYNTHETIC_SEED=7
export SYNTHETIC_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export SYNTHETIC_POOLING=mean
export SYNTHETIC_BATCH_SIZE=512
export SKIP_ARC_SETUP=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=conditioning_minilm_focus
export WANDB_RUN_NAME=synthetic_minilm103424_semantic_fixedval1024_k128_epoch100_long2d
export WANDB_DIR=/data/engs-pnpl/glandau/elf-cache/wandb
export WANDB_DATA_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-data
export WANDB_CACHE_DIR=/data/engs-pnpl/glandau/elf-cache/wandb-cache
export TMPDIR=/data/engs-pnpl/glandau/elf-cache/tmp
mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"

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
if [[ -d "$TMP_ENV_ROOT/bin" && -f "$ENV_STAMP_PATH" && "$(cat "$ENV_STAMP_PATH")" == "$ENV_STAMP" ]]; then
  echo "Skipping ARC env setup; reusing TMP_ENV_ROOT=$TMP_ENV_ROOT ENV_STAMP=$ENV_STAMP"
else
  PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" TMP_ENV_ROOT="$TMP_ENV_ROOT" ENV_STAMP="$ENV_STAMP" \
    bash "$PROJECT_ROOT/scripts/arc_setup_elf_env.sh"
fi
conda activate "$TMP_ENV_ROOT"

if [[ ! -f "$BASE_SYNTHETIC_OUTPUT" ]]; then
  python scripts/export_synthetic_sentence_embeddings.py \
    --output "$BASE_SYNTHETIC_OUTPUT" \
    --count "$SYNTHETIC_COUNT" \
    --seed "$SYNTHETIC_SEED" \
    --embedding-model-name "$SYNTHETIC_EMBEDDING_MODEL" \
    --pooling "$SYNTHETIC_POOLING" \
    --batch-size "$SYNTHETIC_BATCH_SIZE" \
    --device cuda
fi
if [[ ! -f "$FIXEDVAL_SYNTHETIC_OUTPUT" ]]; then
  python scripts/reorder_npz_fixed_validation.py \
    --input-npz "$BASE_SYNTHETIC_OUTPUT" \
    --output "$FIXEDVAL_SYNTHETIC_OUTPUT" \
    --val-start 10240 \
    --val-count 1024
fi

python scripts/train_npz_semantic_to_elf.py \
  --npz-path "$FIXEDVAL_SYNTHETIC_OUTPUT" \
  --use_wandb \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_group "$WANDB_RUN_GROUP" \
  --wandb_run_name "$WANDB_RUN_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --device cuda \
  --num-examples 103424 \
  --val-num-examples 1024 \
  --eval-num-examples 1024 \
  --retrieval-eval-every 8000 \
  --retrieval-num-examples 1024 \
  --retrieval-batch-size 256 \
  --epochs 100 \
  --batch_size 64 \
  --eval_every 4000 \
  --num_sampling_steps 32 \
  --train-timestep-mode sampling_schedule \
  --train-timestep-steps 32 \
  --context_length 128 \
  --semantic-hidden-dim 4096 \
  --last_n_blocks -1 \
  --lr 1e-4 \
  --cfg_scale 1.0 \
  --self_cond_cfg_scale 1.0 \
  --cond_dropout_prob 0.0 \
  --denoiser_loss_weight 1.0 \
  --decoder_loss_weight 1.0 \
  --generation-t5-retrieval \
  --wandb_sample_examples 16 \
  --save-best-checkpoint \
  --seed 7 \
  --wandb_notes "MiniLM pooled combined scale-up: 102400 train examples, same 1024 validation sentences as the 10k run, K=128 bridge, 100 epochs."
