#!/usr/bin/env bash
#SBATCH --job-name=qc4w_pareto
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --array=0-28%6
#SBATCH --output=submit-jobs/screen_qc4wyals_pareto_checkpoints_fixedcache-%A_%a.out
#SBATCH --error=submit-jobs/screen_qc4wyals_pareto_checkpoints_fixedcache-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
run_root=/data/engs-pnpl/glandau/elf-runs
cache=/data/engs-pnpl/glandau/elf-cache/tang_themoth_qc4wyals_ada_v2/interface/qc4wyals_val110_robust_residualaug_adapter_step11250_s32_cfg10_seed49_t5small_latents_fp16.npy

run_dirs=(
  tang_apples_qc4wyals_robust_residualaug_context_preserve_from_robust11250_paretocont_pareto_v1_seed49_20260901
  tang_apples_qc4wyals_robust_residualaug_context_content_from_robust11250_paretocont_pareto_v1_seed49_20260901
  tang_apples_qc4wyals_robust_residualaug_frozenadapter_lora4_from_robust11250_paretocont_pareto_v1_seed49_20260901
  tang_apples_qc4wyals_robust_residualaug_frozenadapter_last2_from_robust11250_paretocont_pareto_v1_seed49_20260901
  tang_apples_qc4wyals_robust_residualaug_context_lora4_from_robust11250_paretocont_pareto_v1_seed49_20260901
)
labels=(ctxp ctxc lora4 last2 ctxlora4)
lora_ranks=(0 0 4 0 4)
step_lists=(
  "1000 1750 2000 6250 8000"
  "250 3000 4750 6000 10000 12000"
  "500 750 2000 4250 8500"
  "1000 1250 6000 7250 9500 12000"
  "1500 2750 3500 4000 7500 7750 11250"
)

candidate_index=$SLURM_ARRAY_TASK_ID
selected_arm=-1
selected_step=
for arm_index in "${!run_dirs[@]}"; do
  read -r -a arm_steps <<< "${step_lists[$arm_index]}"
  if (( candidate_index < ${#arm_steps[@]} )); then
    selected_arm=$arm_index
    selected_step=${arm_steps[$candidate_index]}
    break
  fi
  candidate_index=$((candidate_index - ${#arm_steps[@]}))
done

if (( selected_arm < 0 )) || [[ -z $selected_step ]]; then
  echo "Could not resolve array task $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi

step_padded=$(printf '%08d' "$selected_step")
shopt -s nullglob
matches=( "$run_root/${run_dirs[$selected_arm]}/checkpoints/eval_step_${step_padded}_score_"*.pt )
shopt -u nullglob
if (( ${#matches[@]} != 1 )); then
  printf 'Expected one checkpoint for arm=%s step=%s; found %d\n' \
    "${labels[$selected_arm]}" "$selected_step" "${#matches[@]}" >&2
  printf '%s\n' "${matches[@]:-}" >&2
  exit 2
fi

export SELECTED_CHECKPOINT=${matches[0]}
export SELECTED_TAG=pareto_${labels[$selected_arm]}_step${selected_step}_s32_cfg10_fixedlatent_screen
export SEED_OVERRIDE=49
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CFG_SCALE=1.0
export SELF_COND_CFG_SCALE=1.0
export TARGET_LATENTS_CACHE_OVERRIDE=$cache
export SEMANTIC_ADAPTER_KIND=flat
export ELF_LORA_RANK=${lora_ranks[$selected_arm]}
export ELF_LORA_ALPHA=16
export ELF_LORA_DROPOUT=0.05
export ELF_LORA_LAST_N_BLOCKS=-1

printf 'array_task=%s arm=%s step=%s checkpoint=%s lora_rank=%s\n' \
  "$SLURM_ARRAY_TASK_ID" "${labels[$selected_arm]}" "$selected_step" \
  "$SELECTED_CHECKPOINT" "$ELF_LORA_RANK"

exec bash "$project_root/submit-jobs/eval_qc4wyals_selected_cached_interface.sbatch"
