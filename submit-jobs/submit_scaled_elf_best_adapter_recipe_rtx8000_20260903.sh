#!/usr/bin/env bash
set -euo pipefail

# Maintenance on the L40S fleet blocks the standard scaling ladder.  Run the
# identical experiment on the 48 GB RTX8000 interactive nodes instead.  The
# longer wall time compensates for the older GPU; scientific settings remain
# those in train_elf_semantic_scaling_adapter_ladder.sbatch.

job_id() {
  local value=$1
  printf '%s\n' "${value%%;*}"
}

common=(
  --clusters=htc
  --partition=interactive
  --qos=standard
  --time=12:00:00
  --gres=gpu:rtx8000:1
  --cpus-per-task=8
  --mem=96G
  --export=ALL,RETRIEVAL_NUM_EXAMPLES_OVERRIDE=-1
)

larger=$(job_id "$(sbatch --parsable "${common[@]}" --array=0,3%2 \
  submit-jobs/train_elf_semantic_scaling_adapter_ladder.sbatch)")

controls=$(job_id "$(sbatch --parsable "${common[@]}" --dependency=afterany:$larger \
  --array=4-5%2 submit-jobs/train_elf_semantic_scaling_adapter_ladder.sbatch)")

report=$(job_id "$(sbatch --parsable --clusters=htc --partition=short --qos=standard \
  --time=00:30:00 --dependency=afterany:$controls \
  submit-jobs/report_elf_semantic_scaling_ladder.sbatch)")

printf 'larger_adapters=%s (MRI x4 task 0, MEG x16 task 3)\n' "$larger"
printf 'matched_x1_controls=%s (MRI x1 task 4, MEG x1 task 5)\n' "$controls"
printf 'ladder_report=%s\n' "$report"
