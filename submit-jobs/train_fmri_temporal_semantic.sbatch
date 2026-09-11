#!/usr/bin/env bash
#SBATCH --job-name=fmri_temporal
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=submit-jobs/train_fmri_temporal_semantic-%j.out
#SBATCH --error=submit-jobs/train_fmri_temporal_semantic-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
mri_run=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1
run_tag=${RUN_TAG:-fmri_temporal40tok_d128_l2_exact1_full1_seed49_20260831}
unique_flag=()
if [[ "${UNIQUE_RESPONSE_TRS:-0}" == "1" ]]; then
  unique_flag=(--unique-response-trs)
fi

cd "$project_root"
export PYTHONPATH="$project_root/src:$project_root/scripts:${PYTHONPATH:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$data_root/conda-envs/braindiffusion-elf-torch271cu118"

python scripts/train_fmri_temporal_semantic.py \
  --per-tr-npz "$mri_run/dataset_all_minilm_l6_v2_tr_aligned_delayed.npz" \
  --segment-npz "$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz" \
  --exact-target-npz "$data_root/minilm/knowntext_unseenbrain_v1/oracle_exact_minilm384_train_val.npz" \
  --full-target-npz "$data_root/minilm/temporally_aligned_targets_20260831/full20s/uts01_minilm_l6v2_full20s_train_val_minilm384.npz" \
  --output-dir "/data/engs-pnpl/glandau/elf-runs/$run_tag" \
  --token-dim "${TOKEN_DIM:-128}" \
  --layers "${LAYERS:-2}" \
  --heads "${HEADS:-8}" \
  --exact-weight "${EXACT_WEIGHT:-1.0}" \
  --full-weight "${FULL_WEIGHT:-1.0}" \
  --contrastive-weight "${CONTRASTIVE_WEIGHT:-0.1}" \
  --epochs "${EPOCHS:-30}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --lr "${LR:-2e-4}" \
  --seed "${SEED:-49}" \
  "${unique_flag[@]}"
