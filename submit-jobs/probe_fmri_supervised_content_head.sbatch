#!/usr/bin/env bash
#SBATCH --job-name=fmri_lexprobe
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=priority
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-6
#SBATCH --output=submit-jobs/probe_fmri_supervised_content_head-%A_%a.out
#SBATCH --error=submit-jobs/probe_fmri_supervised_content_head-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
mri_run=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1
native_checkpoint=$mri_run/model.pt
stage_checkpoint=/data/engs-pnpl/glandau/elf-runs/fmri_minilm_stage2_recover2_lora_last2_r4_semantic_strong_from_delaytok_best_ep100_seed49_20260830_from_lora_step3000/checkpoints/eval_step_00001000_score_0.403006.pt
direct384_checkpoint=/data/engs-pnpl/glandau/elf-runs/fmri_minilm_stage2_direct384_semanticonly_exactalign_from_delaytok_best_ep100_seed49_20260830_v100shortprobe_from_best_single_step1000_20260831/semantic_retrieval_step250_snapshot.pt
known_dir=$data_root/minilm/knowntext_unseenbrain_v1
output_dir=/data/engs-pnpl/glandau/elf-runs/fmri_supervised_content_probe_20260831
text_npz=${TEXT_NPZ_OVERRIDE:-$known_dir/oracle_exact_minilm384_train_val.npz}
validation_text_npz=${VALIDATION_TEXT_NPZ_OVERRIDE:-}
extra_args=()
if [[ -n "$validation_text_npz" ]]; then
  extra_args+=(--validation-text-npz "$validation_text_npz")
fi

feature=hidden
head_hidden_dim=512
checkpoint=$native_checkpoint
mri_output_dim=1536
shuffle_train_pairing_seed=-1
case "$SLURM_ARRAY_TASK_ID" in
  0) variant=hidden_mlp512_priorresidual ;;
  1) variant=hidden_linear_priorresidual; head_hidden_dim=0 ;;
  2) variant=semantic1536_mlp512_priorresidual; feature=semantic; checkpoint=$stage_checkpoint ;;
  3) variant=hidden_mlp512_priorresidual_shuffle101; shuffle_train_pairing_seed=101 ;;
  4) variant=hidden_mlp512_priorresidual_shuffle202; shuffle_train_pairing_seed=202 ;;
  5) variant=hidden_mlp512_priorresidual_shuffle303; shuffle_train_pairing_seed=303 ;;
  6) variant=semantic384_directstep250_mlp512_priorresidual; feature=semantic; checkpoint=$direct384_checkpoint; mri_output_dim=384 ;;
  *) echo "Unsupported task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
head_hidden_dim=${HEAD_HIDDEN_DIM_OVERRIDE:-$head_hidden_dim}
variant=${variant}${VARIANT_SUFFIX:-}

mkdir -p "$output_dir"
cd "$project_root"
export PYTHONPATH="$project_root/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$data_root/conda-envs/braindiffusion-elf-torch271cu118}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$TMP_ENV_ROOT"

python scripts/probe_fmri_supervised_content_head.py \
  --brain-npz "$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz" \
  --text-npz "$text_npz" \
  --checkpoint "$checkpoint" \
  --output-json "$output_dir/${variant}.json" \
  --mri-output-dim "$mri_output_dim" \
  --feature "$feature" \
  --head-hidden-dim "$head_hidden_dim" \
  --prior-initialization \
  --shuffle-train-pairing-seed "$shuffle_train_pairing_seed" \
  --min-frequency "${MIN_FREQUENCY_OVERRIDE:-5}" \
  --max-vocabulary "${MAX_VOCABULARY_OVERRIDE:-3000}" \
  --negative-topk 64 \
  --negative-weight "${NEGATIVE_WEIGHT_OVERRIDE:-0.10}" \
  --pairwise-positive-weight "${PAIRWISE_POSITIVE_WEIGHT:-0.0}" \
  --epochs "${EPOCHS_OVERRIDE:-30}" \
  --batch-size 128 \
  --feature-batch-size 256 \
  --lr "${LR_OVERRIDE:-1e-4}" \
  --weight-decay 0.001 \
  --seed 49 \
  "${extra_args[@]}"
