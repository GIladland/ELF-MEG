#!/usr/bin/env bash
#SBATCH --job-name=fmri_delay_audit
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/audit_fmri_delay_content_signal-%j.out
#SBATCH --error=submit-jobs/audit_fmri_delay_content_signal-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
env_root=${ENV_ROOT:-/data/engs-pnpl/glandau/elf-cache/conda-envs/braindiffusion-elf-torch271cu118}
known_dir=/data/engs-pnpl/glandau/elf-cache/minilm/knowntext_unseenbrain_v1
mri_run=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1
checkpoint=/data/engs-pnpl/glandau/elf-runs/fmri_minilm_stage2_recover2_lora_last2_r4_semantic_strong_from_delaytok_best_ep100_seed49_20260830_from_lora_step3000/checkpoints/eval_step_00001000_score_0.403006.pt
output=${AUDIT_OUTPUT:-/data/engs-pnpl/glandau/elf-runs/fmri_delay_content_signal_audit_20260831.json}

cd "$project_root"
"$env_root/bin/python" scripts/audit_fmri_delay_content_signal.py \
  --brain-npz "$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz" \
  --text-npz "$known_dir/oracle_exact_minilm384_train_val.npz" \
  --checkpoint "$checkpoint" \
  --output "$output" \
  --batch-size 128
