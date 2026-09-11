#!/usr/bin/env bash
#SBATCH --job-name=fmri_ordered10
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --output=submit-jobs/probe_fmri_ordered_word_head-%A_%a.out
#SBATCH --error=submit-jobs/probe_fmri_ordered_word_head-%A_%a.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
mri_run=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1
checkpoint=/data/engs-pnpl/glandau/elf-runs/fmri_minilm_stage2_direct384_semanticonly_exactalign_from_delaytok_best_ep100_seed49_20260830_v100shortprobe_from_best_single_step1000_20260831/semantic_retrieval_step250_snapshot.pt
task=${SLURM_ARRAY_TASK_ID:-0}
feature=semantic
hidden=512
pairwise=0.3
content_ce=1.0
lr=3e-4
case "$task" in
  0) variant=semantic_linear_pair0p3; hidden=0; lr=1e-4 ;;
  1) variant=semantic_mlp512_pair0p3 ;;
  2) variant=semantic_mlp512_pair1p0; pairwise=1.0 ;;
  3) variant=hidden2048_mlp512_pair0p3; feature=hidden ;;
  4) variant=semantic_mlp512_pair1p0_contentce2; pairwise=1.0; content_ce=2.0 ;;
  5) variant=semantic_mlp512_pair1p0_contentce3; pairwise=1.0; content_ce=3.0 ;;
  6) variant=semantic_mlp512_pair1p0_contentce5; pairwise=1.0; content_ce=5.0 ;;
  7) variant=semantic_mlp512_pair0p3_contentce3; content_ce=3.0 ;;
  *) echo "Unsupported task $task" >&2; exit 2 ;;
esac
if [[ -n "${EPOCHS_OVERRIDE:-}" ]]; then
  variant=${variant}_ep${EPOCHS_OVERRIDE}_probe
fi
extra_args=()
if [[ -n "${CONTENT_HEAD_CHECKPOINT:-}" ]]; then
  variant=${variant}_contentbias
  extra_args+=(
    --content-head-checkpoint "$CONTENT_HEAD_CHECKPOINT"
    --content-prior-subtraction "${CONTENT_PRIOR_SUBTRACTION:-0.5}"
    --content-bias-strengths 0.25 0.5 1.0 2.0 4.0
  )
fi

output=/data/engs-pnpl/glandau/elf-runs/fmri_ordered_word_probe_20260831/${variant}.json
cd "$project_root"
export PYTHONPATH="$project_root:$project_root/src:$project_root/scripts:${PYTHONPATH:-}"
python_bin=${PYTHON_BIN:-$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python}

"$python_bin" scripts/probe_fmri_ordered_word_head.py \
  --brain-npz "$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz" \
  --text-npz "$data_root/minilm/knowntext_unseenbrain_v1/oracle_exact_minilm384_train_val.npz" \
  --checkpoint "$checkpoint" \
  --output-json "$output" \
  --mri-output-dim 384 \
  --feature "$feature" \
  --head-hidden-dim "$hidden" \
  --pairwise-weight "$pairwise" \
  --content-ce-weight "$content_ce" \
  --max-vocabulary 5000 \
  --min-frequency 2 \
  --derangements "${DERANGEMENTS_OVERRIDE:-10}" \
  --epochs "${EPOCHS_OVERRIDE:-30}" \
  --batch-size 128 \
  --lr "$lr" \
  --seed 49 \
  "${extra_args[@]}"
