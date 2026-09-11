#!/usr/bin/env bash
set -euo pipefail

job_id() {
  local value=$1
  printf '%s\n' "${value%%;*}"
}

# Keep modality failures isolated: a failed ADA preparation must not cancel
# the MiniLM curve, and vice versa.
prep_mri=$(job_id "$(sbatch --parsable --array=0 submit-jobs/prepare_elf_semantic_scaling_corpora.sbatch)")
prep_meg=$(job_id "$(sbatch --parsable --array=1 submit-jobs/prepare_elf_semantic_scaling_corpora.sbatch)")

train_mri=$(job_id "$(sbatch --parsable --dependency=afterok:$prep_mri --array=0-1 submit-jobs/train_elf_semantic_scaling.sbatch)")
train_meg=$(job_id "$(sbatch --parsable --dependency=afterok:$prep_meg --array=2-3 submit-jobs/train_elf_semantic_scaling.sbatch)")

# The existing x1 metrics are already usable. Run their stricter matched
# repeats only after scaled training, so scarce maintenance-window GPUs answer
# the new scaling question first.
scaled_train_deps=$train_mri:$train_meg
x1_exact=$(job_id "$(sbatch --parsable --dependency=afterok:$scaled_train_deps submit-jobs/eval_elf_semantic_scaling_x1_exact.sbatch)")
x1_raw=$(job_id "$(sbatch --parsable --dependency=afterok:$scaled_train_deps --array=4-5 submit-jobs/eval_elf_semantic_scaling_raw_ladder.sbatch)")
x1_adapter=$(job_id "$(sbatch --parsable --dependency=afterok:$scaled_train_deps --array=4-5 submit-jobs/train_elf_semantic_scaling_adapter_ladder.sbatch)")

raw_mri=$(job_id "$(sbatch --parsable --dependency=afterok:$train_mri --array=0-1 submit-jobs/eval_elf_semantic_scaling_raw_ladder.sbatch)")
raw_meg=$(job_id "$(sbatch --parsable --dependency=afterok:$train_meg --array=2-3 submit-jobs/eval_elf_semantic_scaling_raw_ladder.sbatch)")
adapter_mri=$(job_id "$(sbatch --parsable --dependency=afterok:$train_mri --array=0-1 submit-jobs/train_elf_semantic_scaling_adapter_ladder.sbatch)")
adapter_meg=$(job_id "$(sbatch --parsable --dependency=afterok:$train_meg --array=2-3 submit-jobs/train_elf_semantic_scaling_adapter_ladder.sbatch)")

all_jobs=$x1_exact:$x1_raw:$x1_adapter:$train_mri:$train_meg:$raw_mri:$raw_meg:$adapter_mri:$adapter_meg
report=$(job_id "$(sbatch --parsable --dependency=afterany:$all_jobs submit-jobs/report_elf_semantic_scaling_ladder.sbatch)")

printf 'prep_mri=%s\n' "$prep_mri"
printf 'prep_meg=%s\n' "$prep_meg"
printf 'x1_exact=%s\n' "$x1_exact"
printf 'x1_raw=%s\n' "$x1_raw"
printf 'x1_adapter=%s\n' "$x1_adapter"
printf 'train_mri=%s\n' "$train_mri"
printf 'train_meg=%s\n' "$train_meg"
printf 'raw_mri=%s\n' "$raw_mri"
printf 'raw_meg=%s\n' "$raw_meg"
printf 'adapter_mri=%s\n' "$adapter_mri"
printf 'adapter_meg=%s\n' "$adapter_meg"
printf 'report=%s\n' "$report"
