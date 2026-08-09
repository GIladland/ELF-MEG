#!/usr/bin/env bash
#SBATCH --job-name=t5pool_e1000
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/synthetic_t5pooled1024_semantic_full_epoch1000_long2d-%j.out
#SBATCH --error=submit-jobs/synthetic_t5pooled1024_semantic_full_epoch1000_long2d-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/synthetic_t5pooled1024_semantic_full_epoch1000_long2d
export SYNTHETIC_T5_OUTPUT=/data/engs-pnpl/glandau/elf-cache/synthetic/synthetic_t5_1024_seed7.npz
export T5_POOLED_OUTPUT=/data/engs-pnpl/glandau/elf-cache/synthetic/synthetic_t5pooled_1024_seed7.npz
export SYNTHETIC_COUNT=1024
export SYNTHETIC_SEED=7
export SKIP_ARC_SETUP=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=conditioning_ladder
export WANDB_RUN_NAME=synthetic_t5pooled1024_semantic_full_epoch1000_long2d
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

if [[ ! -f "$SYNTHETIC_T5_OUTPUT" ]]; then
  python scripts/export_synthetic_t5_elf_npz.py \
    --output "$SYNTHETIC_T5_OUTPUT" \
    --count "$SYNTHETIC_COUNT" \
    --seed "$SYNTHETIC_SEED" \
    --batch-size 256 \
    --device cuda
else
  echo "Reusing synthetic T5 NPZ: $SYNTHETIC_T5_OUTPUT"
fi

if [[ ! -f "$T5_POOLED_OUTPUT" ]]; then
  python scripts/export_t5_pooled_semantic_npz.py \
    --input-npz "$SYNTHETIC_T5_OUTPUT" \
    --output "$T5_POOLED_OUTPUT"
else
  echo "Reusing pooled T5 NPZ: $T5_POOLED_OUTPUT"
fi

python scripts/train_npz_semantic_to_elf.py \
  --npz-path "$T5_POOLED_OUTPUT" \
  --use_wandb \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_group "$WANDB_RUN_GROUP" \
  --wandb_run_name "$WANDB_RUN_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --device cuda \
  --num-examples 1024 \
  --eval-num-examples 128 \
  --retrieval-eval-every 1000 \
  --retrieval-num-examples 1024 \
  --retrieval-batch-size 512 \
  --steps 16000 \
  --batch_size 64 \
  --eval_every 500 \
  --num_sampling_steps 32 \
  --train-timestep-mode sampling_schedule \
  --train-timestep-steps 32 \
  --context_length 20 \
  --semantic-hidden-dim 2048 \
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
  --wandb_notes "Pooled T5 control: masked mean pool of the same ELF-normalized T5 latents used by the T5 identity run, giving one 512-d vector per sentence. Trained with the same K=20 semantic-vector projector and 16000-step schedule as the pooled MiniLM/MPNet/E5 runs."
