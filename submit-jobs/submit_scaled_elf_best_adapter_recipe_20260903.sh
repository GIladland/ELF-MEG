#!/usr/bin/env bash
set -euo pipefail

# Apply the validated modality-specific brain-to-ELF recipes to the completed
# scaled diffusion checkpoints.  The larger checkpoints run first; matched x1
# controls then repeat the exact same code path for a clean scale comparison.
# htc-g083 is outside the active maintenance reservation and currently has two
# free L40S GPUs.

job_id() {
  local value=$1
  printf '%s\n' "${value%%;*}"
}

common=(
  --clusters=htc
  --partition=short
  --qos=standard
  --nodelist=htc-g083
  --time=06:00:00
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
