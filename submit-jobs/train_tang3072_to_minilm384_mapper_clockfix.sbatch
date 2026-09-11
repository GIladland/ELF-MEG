#!/usr/bin/env bash
#SBATCH --job-name=tang2minilm
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=submit-jobs/train_tang3072_to_minilm384_mapper_clockfix-%j.out
#SBATCH --error=submit-jobs/train_tang3072_to_minilm384_mapper_clockfix-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
source_dir=$DATA_ROOT/tang_gpt/oracle3072_middle10content_clockfix
source_prefix=uts01_tang_gpt_lanczos_seg10_oracle3072_middle10content_clockfix
target_dir=$DATA_ROOT/tang_gpt/oracle3072_middle10content_clockfix_minilm
mapper_variant=${MAPPER_VARIANT:-mlp}
hidden_dim=${HIDDEN_DIM:-1024}
mapper_dropout=${MAPPER_DROPOUT:-0.1}
output_dir=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/tang3072_to_minilm384_center10_clockfix_${mapper_variant}_20260828}

export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export TORCH_PIP_SPEC=${TORCH_PIP_SPEC:-torch==2.7.1}
export TORCH_PIP_INDEX_URL=${TORCH_PIP_INDEX_URL:-https://download.pytorch.org/whl/cu118}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$DATA_ROOT/xdg-cache}
export HF_HOME=${HF_HOME:-$DATA_ROOT/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$DATA_ROOT/pip-cache}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$DATA_ROOT/conda-pkgs}
export WANDB_DIR=${WANDB_DIR:-$DATA_ROOT/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$DATA_ROOT/wandb-cache}
export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}

source "$(conda info --base)/etc/profile.d/conda.sh"
stamp_path=$TMP_ENV_ROOT/.elf_env_stamp
if [[ ! -d "$TMP_ENV_ROOT/bin" || ! -f "$stamp_path" || "$(cat "$stamp_path")" != "$ENV_STAMP" ]]; then
  bash "$PROJECT_ROOT/scripts/arc_setup_elf_env.sh"
fi
conda activate "$TMP_ENV_ROOT"
cd "$PROJECT_ROOT"
mkdir -p "$output_dir" "$WANDB_DIR" "$WANDB_CACHE_DIR"

python scripts/train_semantic_mapper.py \
  --source-train-val "$source_dir/${source_prefix}_train_val.npz" \
  --target-train-val "$target_dir/${source_prefix}_train_val_minilm.npz" \
  --source-test "$source_dir/${source_prefix}_test.npz" \
  --target-test "$target_dir/${source_prefix}_test_minilm.npz" \
  --source-predicted-test "$source_dir/uts01_lanczos_seg10_test_pred3072_middle10content_clockfix.npz" \
  --output-dir "$output_dir" \
  --val-count 266 \
  --hidden-dim "$hidden_dim" \
  --dropout "$mapper_dropout" \
  --epochs 200 \
  --batch-size 256 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --contrastive-weight 0.1 \
  --temperature 0.07 \
  --patience 30 \
  --device cuda \
  --use-wandb \
  --wandb-project BrainDiffusion \
  --wandb-entity giladland-university-of-oxford \
  --wandb-group tang3072_to_minilm384_clockfix_20260828 \
  --wandb-run-name "tang3072_to_minilm384_center10_clockfix_${mapper_variant}_20260828" \
  --wandb-notes "Supervised semantic-space rescue ($mapper_variant): native oracle Tang-GPT 3072 to exact centered-10 MiniLM-384, trained on 11725 rows and validated on the held-out 266-row story split. Exports oracle and MRI2SEM-predicted test embeddings for the pretrained MiniLM-to-ELF bridge."
