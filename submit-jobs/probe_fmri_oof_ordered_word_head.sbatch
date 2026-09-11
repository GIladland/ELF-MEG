#!/usr/bin/env bash
#SBATCH --job-name=fmri_ooford10
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=72G
#SBATCH --output=submit-jobs/probe_fmri_oof_ordered_word_head-%A_%a.out
#SBATCH --error=submit-jobs/probe_fmri_oof_ordered_word_head-%A_%a.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
mri_run=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1
semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz
checkpoint=/data/engs-pnpl/glandau/elf-runs/fmri_minilm_stage2_direct384_semanticonly_exactalign_from_delaytok_best_ep100_seed49_20260830_v100shortprobe_from_best_single_step1000_20260831/semantic_retrieval_step250_snapshot.pt
content=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt
task=${SLURM_ARRAY_TASK_ID:-0}
hidden=0
pairwise=0.3
content_checkpoint=
case "$task" in
  0) variant=oofmean_linear_pair0p3 ;;
  1) variant=oofmean_linear_pair0p3_contentbias; content_checkpoint=$content ;;
  2) variant=oofmean_mlp512_pair1p0_contentbias; hidden=512; pairwise=1.0; content_checkpoint=$content ;;
  *) exit 2 ;;
esac
if [[ -n "${EPOCHS_OVERRIDE:-}" ]]; then
  variant=${variant}_ep${EPOCHS_OVERRIDE}_probe
fi
content_args=()
if [[ -n "$content_checkpoint" ]]; then
  content_args=(
    --content-head-checkpoint "$content_checkpoint"
    --content-prior-subtraction 0.5
    --content-bias-strengths 0.25 0.5 1.0 2.0
  )
fi
output=/data/engs-pnpl/glandau/elf-runs/fmri_oof_ordered_word_probe_20260831/${variant}.json

cd "$project_root"
export PYTHONPATH="$project_root:$project_root/src:$project_root/scripts:${PYTHONPATH:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$data_root/conda-envs/braindiffusion-elf-torch271cu118"

python scripts/probe_fmri_ordered_word_head.py \
  --brain-npz "$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz" \
  --text-npz "$data_root/minilm/knowntext_unseenbrain_v1/oracle_exact_minilm384_train_val.npz" \
  --semantic-npz "$semantic" \
  --checkpoint "$checkpoint" \
  --output-json "$output" \
  --feature semantic \
  --head-hidden-dim "$hidden" \
  --pairwise-weight "$pairwise" \
  --max-vocabulary 5000 \
  --min-frequency 2 \
  --derangements 10 \
  --epochs "${EPOCHS_OVERRIDE:-30}" \
  --batch-size 128 \
  --lr 1e-4 \
  --seed 49 \
  "${content_args[@]}"
