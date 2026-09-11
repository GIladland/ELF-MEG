#!/usr/bin/env bash
#SBATCH --job-name=fmri_time_audit
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=submit-jobs/audit_fmri_temporal_semantic_targets-%j.out
#SBATCH --error=submit-jobs/audit_fmri_temporal_semantic_targets-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
mri_run=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1
checkpoint=/data/engs-pnpl/glandau/elf-runs/fmri_minilm_stage2_direct384_semanticonly_exactalign_from_delaytok_best_ep100_seed49_20260830_v100shortprobe_from_best_single_step1000_20260831/semantic_retrieval_step250_snapshot.pt
known_dir=$data_root/minilm/knowntext_unseenbrain_v1
temporal_root=$data_root/minilm/temporally_aligned_targets_20260831
output=/data/engs-pnpl/glandau/elf-runs/fmri_temporal_target_audit_20260831/direct384_step250_val266.json

cd "$project_root"
export PYTHONPATH="$project_root/src:${PYTHONPATH:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$data_root/conda-envs/braindiffusion-elf-torch271cu118"

python scripts/audit_fmri_temporal_semantic_targets.py \
  --brain-npz "$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz" \
  --checkpoint "$checkpoint" \
  --output-json "$output" \
  --target "content10=$known_dir/oracle_exact_minilm384_train_val.npz" \
  --target "centered10=$temporal_root/centered10/uts01_minilm_l6v2_centered10_train_val_minilm384.npz" \
  --target "full20s=$temporal_root/full20s/uts01_minilm_l6v2_full20s_train_val_minilm384.npz"
