#!/usr/bin/env bash
#SBATCH --job-name=qc4w_mapper
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4%5
#SBATCH --output=submit-jobs/train_qc4wyals_ada_mapper_matrix-%A_%a.out
#SBATCH --error=submit-jobs/train_qc4wyals_ada_mapper_matrix-%A_%a.err

set -euo pipefail

project_root=/data/engs-pnpl/glandau/BrainDiffusion/ELF
data_root=/data/engs-pnpl/glandau/elf-cache/tang_themoth_qc4wyals_ada_v2/interface
python_bin=/data/engs-pnpl/glandau/elf-cache/conda-envs/braindiffusion-elf-torch271cu118/bin/python

hidden_dim=0
dropout=0.0
lr=3e-4
weight_decay=1e-4
contrastive_weight=0.10
residual_arg=()

case "$SLURM_ARRAY_TASK_ID" in
  0)
    arm=linear
    ;;
  1)
    arm=mlp512
    hidden_dim=512
    dropout=0.10
    ;;
  2)
    arm=reslinear
    residual_arg+=(--residual)
    lr=1e-4
    ;;
  3)
    arm=resmlp512
    hidden_dim=512
    dropout=0.10
    residual_arg+=(--residual)
    lr=1e-4
    ;;
  4)
    arm=resmlp1536
    hidden_dim=1536
    dropout=0.10
    residual_arg+=(--residual)
    lr=1e-4
    ;;
  *) exit 2 ;;
esac

mapper_epochs=${MAPPER_EPOCHS:-400}
tag_suffix=${MAPPER_TAG_SUFFIX:-full400}
output_dir=/data/engs-pnpl/glandau/elf-runs/qc4wyals_ada_mapper_${arm}_${tag_suffix}_seed49_20260901

cd "$project_root"
exec "$python_bin" scripts/train_semantic_mapper.py \
  --source-train-val "$data_root/qc4wyals_predicted_train_valtail_ada002.npz" \
  --target-train-val "$data_root/qc4wyals_exact_alignment_train_valtail_ada002.npz" \
  --source-test "$data_root/qc4wyals_test_predicted_ada002.npz" \
  --target-test "$data_root/qc4wyals_test_exact_ada002.npz" \
  --output-dir "$output_dir" \
  --val-count 110 \
  --hidden-dim "$hidden_dim" \
  --dropout "$dropout" \
  --epochs "$mapper_epochs" \
  --batch-size 128 \
  --lr "$lr" \
  --weight-decay "$weight_decay" \
  --contrastive-weight "$contrastive_weight" \
  --temperature 0.07 \
  --patience 60 \
  --eval-every 1 \
  --seed 49 \
  --device cuda \
  --use-wandb \
  --wandb-project BrainDiffusion \
  --wandb-entity giladland-university-of-oxford \
  --wandb-group tang_apples_qc4wyals_ada_mapper_20260901 \
  --wandb-run-name "qc4wyals_ada_mapper_${arm}_${tag_suffix}_seed49_20260901" \
  --wandb-notes "Leakage-safe qc4wyals predicted ADA to exact ADA mapper. Train=2652, selection=held-out val110, final BirthOfANation test26 never used for optimization or selection." \
  "${residual_arg[@]}"
