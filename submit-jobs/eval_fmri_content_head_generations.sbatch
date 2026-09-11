#!/usr/bin/env bash
#SBATCH --job-name=fmri_lexdecode
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=submit-jobs/eval_fmri_content_head_generations-%j.out
#SBATCH --error=submit-jobs/eval_fmri_content_head_generations-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
mri_run=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1
direct_run=/data/engs-pnpl/glandau/elf-runs/fmri_minilm_stage2_direct384_semanticonly_exactalign_from_delaytok_best_ep100_seed49_20260830_v100shortprobe_from_best_single_step1000_20260831
probe_dir=/data/engs-pnpl/glandau/elf-runs/fmri_supervised_content_probe_20260831
output_dir=/data/engs-pnpl/glandau/elf-runs/fmri_content_head_generation_audit_20260831
head_checkpoint=${HEAD_CHECKPOINT_OVERRIDE:-$probe_dir/semantic384_directstep250_mlp512_priorresidual.pt}
output_json=${OUTPUT_JSON_OVERRIDE:-$output_dir/direct384_step250_top5_val266.json}
trainbank_retrieval=${TRAINBANK_RETRIEVAL:-0}
extra_args=()
if [[ -n "${PRIOR_SUBTRACTION:-}" ]]; then
  extra_args+=(--prior-subtraction "$PRIOR_SUBTRACTION")
fi
if [[ -n "${PRIOR_TEXT_NPZ:-}" ]]; then
  extra_args+=(--prior-text-npz "$PRIOR_TEXT_NPZ")
fi
if [[ "$trainbank_retrieval" == "1" ]]; then
  extra_args+=(--trainbank-retrieval --retrieval-query-topk "${RETRIEVAL_QUERY_TOPK:-20}")
fi

cd "$project_root"
export PYTHONPATH="$project_root/src:$project_root/scripts:${PYTHONPATH:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$data_root/conda-envs/braindiffusion-elf-torch271cu118"

python scripts/eval_fmri_content_head_generations.py \
  --brain-npz "$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz" \
  --text-npz "$data_root/minilm/knowntext_unseenbrain_v1/oracle_exact_minilm384_train_val.npz" \
  --mri-checkpoint "$direct_run/semantic_retrieval_step250_snapshot.pt" \
  --head-checkpoint "$head_checkpoint" \
  --output-json "$output_json" \
  --mri-output-dim 384 \
  --topk "${TOPK_OVERRIDE:-5}" \
  --shuffle-repeats 20 \
  "${extra_args[@]}"
