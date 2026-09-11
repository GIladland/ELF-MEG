#!/usr/bin/env bash
#SBATCH --job-name=fmri_ooflex
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=submit-jobs/probe_fmri_oof_content_head-%A_%a.out
#SBATCH --error=submit-jobs/probe_fmri_oof_content_head-%A_%a.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
mri_run=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1
text=${TEXT_NPZ_OVERRIDE:-$data_root/minilm/knowntext_unseenbrain_v1/oracle_exact_minilm384_train_val.npz}
validation_text=${VALIDATION_TEXT_NPZ_OVERRIDE:-}
temporal_context_offsets=${TEMPORAL_CONTEXT_OFFSETS:-0}
positive_idf_power=${POSITIVE_IDF_POWER:-0}
selection_metric=${SELECTION_METRIC:-f1}
semantic_delay_mode=${SEMANTIC_DELAY_MODE:-mean}
shuffle_train_pairing_seed=${SHUFFLE_TRAIN_PAIRING_SEED:--1}
semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz
checkpoint=/data/engs-pnpl/glandau/elf-runs/fmri_minilm_stage2_direct384_semanticonly_exactalign_from_delaytok_best_ep100_seed49_20260830_v100shortprobe_from_best_single_step1000_20260831/semantic_retrieval_step250_snapshot.pt
task=${SLURM_ARRAY_TASK_ID:-0}
hidden=512
pair=0.3
case "$task" in
  0) variant=oofmean_mlp512_pair0p3 ;;
  1) variant=oofmean_mlp512_pair1p0; pair=1.0 ;;
  2) variant=oofmean_linear_pair1p0; hidden=0; pair=1.0 ;;
  *) exit 2 ;;
esac
variant=${variant}${VARIANT_SUFFIX:-}
output=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/${variant}.json
validation_args=()
if [[ -n "$validation_text" ]]; then
  validation_args=(--validation-text-npz "$validation_text")
fi

cd "$project_root"
export PYTHONPATH="$project_root:$project_root/src:$project_root/scripts:${PYTHONPATH:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$data_root/conda-envs/braindiffusion-elf-torch271cu118"

python scripts/probe_fmri_supervised_content_head.py \
  --brain-npz "$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz" \
  --text-npz "$text" \
  --semantic-npz "$semantic" \
  --semantic-delay-mode "$semantic_delay_mode" \
  --checkpoint "$checkpoint" \
  --output-json "$output" \
  --feature semantic \
  --head-hidden-dim "$hidden" \
  --prior-initialization \
  --shuffle-train-pairing-seed "$shuffle_train_pairing_seed" \
  --pairwise-positive-weight "$pair" \
  --temporal-context-offset-trs="$temporal_context_offsets" \
  --positive-idf-power "$positive_idf_power" \
  --selection-metric "$selection_metric" \
  --epochs 20 \
  --batch-size 128 \
  --seed 49 \
  "${validation_args[@]}"
