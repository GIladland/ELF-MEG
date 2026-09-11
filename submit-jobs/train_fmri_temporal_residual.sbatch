#!/usr/bin/env bash
#SBATCH --job-name=fmri_temp_res
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=submit-jobs/train_fmri_temporal_residual-%A_%a.out
#SBATCH --error=submit-jobs/train_fmri_temporal_residual-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
mri_run=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1
base_checkpoint=/data/engs-pnpl/glandau/elf-runs/fmri_minilm_stage2_direct384_semanticonly_exactalign_from_delaytok_best_ep100_seed49_20260830_v100shortprobe_from_best_single_step1000_20260831/semantic_retrieval_step250_snapshot.pt
task=${SLURM_ARRAY_TASK_ID:-0}

unique_flag=()
token_dim=64
exact_weight=1.0
exact_weight_start=1.0
full_weight=0.5
full_weight_end=0.5
case "$task" in
  0)
    variant=overlap40tok_fixedfull0p5_d64
    ;;
  1)
    variant=unique13tr_fixedfull0p5_d64
    unique_flag=(--unique-response-trs)
    ;;
  2)
    variant=unique13tr_curriculum_exact0p25to1_full1to0p1_d64
    unique_flag=(--unique-response-trs)
    exact_weight_start=0.25
    full_weight=1.0
    full_weight_end=0.1
    ;;
  3)
    variant=unique13tr_exactonly_d128
    unique_flag=(--unique-response-trs)
    token_dim=128
    full_weight=0.0
    full_weight_end=0.0
    ;;
  *)
    echo "Unsupported task $task" >&2
    exit 2
    ;;
esac

run_tag=fmri_temporal_residual_${variant}_seed49_20260831
cd "$project_root"
export PYTHONPATH="$project_root:$project_root/src:$project_root/scripts:${PYTHONPATH:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$data_root/conda-envs/braindiffusion-elf-torch271cu118"

python scripts/train_fmri_temporal_residual.py \
  --per-tr-npz "$mri_run/dataset_all_minilm_l6_v2_tr_aligned_delayed.npz" \
  --segment-npz "$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz" \
  --exact-target-npz "$data_root/minilm/knowntext_unseenbrain_v1/oracle_exact_minilm384_train_val.npz" \
  --full-target-npz "$data_root/minilm/temporally_aligned_targets_20260831/full20s/uts01_minilm_l6v2_full20s_train_val_minilm384.npz" \
  --base-checkpoint "$base_checkpoint" \
  --output-dir "/data/engs-pnpl/glandau/elf-runs/$run_tag" \
  --token-dim "$token_dim" \
  --layers 2 \
  --heads 8 \
  --exact-weight "$exact_weight" \
  --exact-weight-start "$exact_weight_start" \
  --full-weight "$full_weight" \
  --full-weight-end "$full_weight_end" \
  --contrastive-weight 0.1 \
  --residual-l2-weight 0.01 \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-4 \
  --seed 49 \
  "${unique_flag[@]}"
