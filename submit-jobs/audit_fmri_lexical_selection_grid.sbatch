#!/usr/bin/env bash
#SBATCH --job-name=fmri_lex_grid
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/audit_fmri_lexical_selection_grid-%j.out
#SBATCH --error=submit-jobs/audit_fmri_lexical_selection_grid-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
environment_root=${TMP_ENV_ROOT:-$data_root/conda-envs/braindiffusion-elf-torch271cu118}
export PATH="$environment_root/bin:$PATH"
export PYTHONNOUSERSITE=1
cd "$project_root"

secondary_args=()
if [[ -n ${SECONDARY_LEXICAL_CHECKPOINT:-} ]]; then
  secondary_args+=(--secondary-lexical-checkpoint "$SECONDARY_LEXICAL_CHECKPOINT")
  secondary_args+=(--secondary-mix "${SECONDARY_MIX:-0,0.1,0.25,0.5}")
  if [[ -n ${SECONDARY_TEXT_NPZ:-} ]]; then
    secondary_args+=(--secondary-text-npz "$SECONDARY_TEXT_NPZ")
  fi
fi

python scripts/audit_fmri_lexical_selection_grid.py \
  --brain-npz /data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz \
  --text-npz "$data_root/minilm/knowntext_unseenbrain_v1/oracle_exact_minilm384_train_val.npz" \
  --mri-checkpoint /data/engs-pnpl/glandau/elf-runs/fmri_minilm_stage2_recover2_lora_last2_r4_semantic_strong_from_delaytok_best_ep100_seed49_20260830_from_lora_step3000/checkpoints/eval_step_00001000_score_0.403006.pt \
  --lexical-checkpoint /data/engs-pnpl/glandau/elf-runs/fmri_supervised_content_probe_20260831/semantic1536_mlp512_priorresidual.pt \
  "${secondary_args[@]}" \
  --output-json "${OUTPUT_JSON:-/data/engs-pnpl/glandau/elf-runs/fmri_supervised_content_probe_20260831/semantic1536_selection_grid_val266.json}"
