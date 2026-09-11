#!/usr/bin/env bash
#SBATCH --job-name=dmini2exact384
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-1
#SBATCH --output=submit-jobs/train_delayed_minilm1536_to_exact_minilm384_mapper-%A_%a.out
#SBATCH --error=submit-jobs/train_delayed_minilm1536_to_exact_minilm384_mapper-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
source_dir=$DATA_ROOT/minilm/oracle1536_delayed_seg10_middle10content_clockfix
target_dir=$DATA_ROOT/tang_gpt/oracle3072_middle10content_clockfix_minilm

case "$SLURM_ARRAY_TASK_ID" in
  0) mapper_variant=mlp1024; hidden_dim=1024; mapper_dropout=0.1 ;;
  1) mapper_variant=linear; hidden_dim=0; mapper_dropout=0.0 ;;
  *) echo "Unsupported task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

run_tag=delayed_minilm1536_to_exact_center10_minilm384_${mapper_variant}_train11725_val266_20260829
output_dir=/data/engs-pnpl/glandau/elf-runs/$run_tag

export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
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
  --source-train-val "$source_dir/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_oracle1536_middle10content_clockfix_train_val.npz" \
  --target-train-val "$target_dir/uts01_tang_gpt_lanczos_seg10_oracle3072_middle10content_clockfix_train_val_minilm.npz" \
  --source-test "$source_dir/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_oracle1536_middle10content_clockfix_test.npz" \
  --target-test "$target_dir/uts01_tang_gpt_lanczos_seg10_oracle3072_middle10content_clockfix_test_minilm.npz" \
  --source-predicted-test "$source_dir/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_pred1536_test_middle10content_clockfix.npz" \
  --output-dir "$output_dir" \
  --allow-sentence-mismatch \
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
  --wandb-group delayed_minilm1536_to_exact_minilm384_20260829 \
  --wandb-run-name "$run_tag" \
  --wandb-notes "Metadata-aligned downstream semantic mapper from delayed MiniLM1536 to exact center-anchored MiniLM384. Trained on 11725 oracle rows and selected on 266 validation rows. Source/target sentences intentionally differ because their documented 10-word timing contracts differ; story and TR boundaries match exactly. The MRI2SEM brain decoder and 107 fMRI test predictions remain fixed."
