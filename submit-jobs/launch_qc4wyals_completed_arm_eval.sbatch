#!/usr/bin/env bash
#SBATCH --job-name=qc4w_launch5
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=submit-jobs/launch_qc4wyals_completed_arm_eval-%j.out
#SBATCH --error=submit-jobs/launch_qc4wyals_completed_arm_eval-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
: "${RUN_GLOB:?Set RUN_GLOB to exactly one completed training run directory glob}"
: "${SELECTED_TAG:?Set SELECTED_TAG to a unique validation evaluation tag}"

semantic_adapter_kind=${SEMANTIC_ADAPTER_KIND:-flat}
elf_lora_rank=${ELF_LORA_RANK:-0}
elf_lora_alpha=${ELF_LORA_ALPHA:-16}
elf_lora_dropout=${ELF_LORA_DROPOUT:-0.05}
elf_lora_last_n_blocks=${ELF_LORA_LAST_N_BLOCKS:--1}
num_sampling_steps=${NUM_SAMPLING_STEPS:-32}
cfg_scale=${CFG_SCALE:-1.0}
self_cond_cfg_scale=${SELF_COND_CFG_SCALE:-1.0}

shopt -s nullglob
matches=( $RUN_GLOB )
shopt -u nullglob
if (( ${#matches[@]} != 1 )); then
  printf 'Expected exactly one run for RUN_GLOB=%s; found %d\n' "$RUN_GLOB" "${#matches[@]}" >&2
  printf '%s\n' "${matches[@]:-}" >&2
  exit 2
fi

checkpoint=${matches[0]}/best.pt
[[ -f "$checkpoint" ]] || { echo "Missing selected checkpoint: $checkpoint" >&2; exit 2; }

if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf 'dry_run=1\ncheckpoint=%s\nselected_tag=%s\nsemantic_adapter_kind=%s\nelf_lora_rank=%s\nnum_sampling_steps=%s\ncfg_scale=%s\n' \
    "$checkpoint" "$SELECTED_TAG" "$semantic_adapter_kind" "$elf_lora_rank" \
    "$num_sampling_steps" "$cfg_scale"
  exit 0
fi

cd "$project_root"
eval_submission=$(sbatch --parsable --clusters=htc --array=0-4%4 \
  --export=ALL,SELECTED_CHECKPOINT="$checkpoint",SELECTED_TAG="$SELECTED_TAG",SEMANTIC_ADAPTER_KIND="$semantic_adapter_kind",ELF_LORA_RANK="$elf_lora_rank",ELF_LORA_ALPHA="$elf_lora_alpha",ELF_LORA_DROPOUT="$elf_lora_dropout",ELF_LORA_LAST_N_BLOCKS="$elf_lora_last_n_blocks",NUM_SAMPLING_STEPS="$num_sampling_steps",TRAIN_TIMESTEP_STEPS="$num_sampling_steps",CFG_SCALE="$cfg_scale",SELF_COND_CFG_SCALE="$self_cond_cfg_scale" \
  submit-jobs/eval_qc4wyals_selected_cached_interface.sbatch)
eval_job=${eval_submission%%;*}

bert_submission=$(sbatch --parsable --clusters=htc --dependency=afterok:"$eval_job" --array=0-4%5 \
  --export=ALL,SELECTED_TAG="$SELECTED_TAG" \
  submit-jobs/eval_qc4wyals_multiseed_bertscore.sbatch)
bert_job=${bert_submission%%;*}

summary_submission=$(sbatch --parsable --clusters=htc --dependency=afterok:"$bert_job" \
  --export=ALL,SELECTED_TAG="$SELECTED_TAG" \
  submit-jobs/summarize_qc4wyals_multiseed.sbatch)
summary_job=${summary_submission%%;*}

printf 'checkpoint=%s\neval_job=%s\nbert_job=%s\nsummary_job=%s\n' \
  "$checkpoint" "$eval_job" "$bert_job" "$summary_job"
