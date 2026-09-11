#!/usr/bin/env bash
#SBATCH --job-name=fmri_oof_mapper
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --array=0-1
#SBATCH --output=submit-jobs/train_fmri_minilm_knowntext_oof_mapper-%A_%a.out
#SBATCH --error=submit-jobs/train_fmri_minilm_knowntext_oof_mapper-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export WANDB_DIR=${WANDB_DIR:-$DATA_ROOT/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$DATA_ROOT/wandb-cache}

source_dir=$DATA_ROOT/minilm/oracle1536_delayed_seg10_middle10content_clockfix
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
oof_source=$DATA_ROOT/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz
oof_target=$known_dir/oofmix_exact_minilm384_targets_train23450_val266.npz
oracle_test=$source_dir/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_oracle1536_middle10content_clockfix_test.npz
predicted_test=$source_dir/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_pred1536_test_middle10content_clockfix.npz
exact_test=$known_dir/oracle_exact_minilm384_test107.npz

case "$SLURM_ARRAY_TASK_ID" in
  0) variant=linear; hidden_dim=0; dropout=0.0 ;;
  1) variant=mlp1024; hidden_dim=1024; dropout=0.1 ;;
  *) exit 2 ;;
esac

run_tag=fmri_minilm_knowntext_oofmix1536_to_exact384_${variant}_train23450_val266_20260829
output_dir=/data/engs-pnpl/glandau/elf-runs/$run_tag

source "$(conda info --base)/etc/profile.d/conda.sh"
stamp_path=$TMP_ENV_ROOT/.elf_env_stamp
if [[ ! -d "$TMP_ENV_ROOT/bin" || ! -f "$stamp_path" || "$(cat "$stamp_path")" != "$ENV_STAMP" ]]; then
  bash "$PROJECT_ROOT/scripts/arc_setup_elf_env.sh"
fi
conda activate "$TMP_ENV_ROOT"
cd "$PROJECT_ROOT"
mkdir -p "$output_dir" "$WANDB_DIR" "$WANDB_CACHE_DIR"

python scripts/train_semantic_mapper.py \
  --source-train-val "$oof_source" \
  --target-train-val "$oof_target" \
  --source-test "$oracle_test" \
  --target-test "$exact_test" \
  --source-predicted-test "$predicted_test" \
  --output-dir "$output_dir" \
  --val-count 266 \
  --hidden-dim "$hidden_dim" \
  --dropout "$dropout" \
  --epochs 200 \
  --batch-size 256 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --contrastive-weight 0.1 \
  --temperature 0.07 \
  --patience 30 \
  --device cuda \
  --seed 49 \
  --use-wandb \
  --wandb-project BrainDiffusion \
  --wandb-entity giladland-university-of-oxford \
  --wandb-group fmri_minilm_knowntext_unseenbrain_mapper_20260829 \
  --wandb-run-name "$run_tag" \
  --wandb-notes "Leakage-safe $variant mapper from delayed MiniLM1536 to exact sentence MiniLM384. Training contains equal oracle and five-fold story-OOF MRI2SEM rows; checkpoint selection uses only the separate 266-row predicted-validation ensemble. The 107 test MRI2SEM vectors are used only for the final post-training mapper evaluation."

