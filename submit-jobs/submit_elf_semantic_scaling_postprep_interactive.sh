#!/usr/bin/env bash
set -euo pipefail

# Resume the scaling ladder after both audited corpora have been prepared.
# The interactive HTC nodes are the only currently available GPUs during the
# maintenance reservation.  Stage-1 training uses ~12k updates and is followed
# by matched raw-brain and trainable-adapter evaluations.

job_id() {
  local value=$1
  printf '%s\n' "${value%%;*}"
}

gpu_args=(--clusters=htc --partition=interactive --qos=standard --gres=gpu:rtx8000:1)

train_mri=$(job_id "$(sbatch --parsable "${gpu_args[@]}" --time=12:00:00 --array=0-1%2 submit-jobs/train_elf_semantic_scaling.sbatch)")
train_meg=$(job_id "$(sbatch --parsable "${gpu_args[@]}" --time=12:00:00 --array=2-3%2 submit-jobs/train_elf_semantic_scaling.sbatch)")

raw_mri=$(job_id "$(sbatch --parsable "${gpu_args[@]}" --time=04:00:00 --dependency=afterok:$train_mri --array=0-1%2 submit-jobs/eval_elf_semantic_scaling_raw_ladder.sbatch)")
raw_meg=$(job_id "$(sbatch --parsable "${gpu_args[@]}" --time=04:00:00 --dependency=afterok:$train_meg --array=2-3%2 submit-jobs/eval_elf_semantic_scaling_raw_ladder.sbatch)")
adapter_mri=$(job_id "$(sbatch --parsable "${gpu_args[@]}" --time=08:00:00 --dependency=afterok:$train_mri --array=0-1%2 submit-jobs/train_elf_semantic_scaling_adapter_ladder.sbatch)")
adapter_meg=$(job_id "$(sbatch --parsable "${gpu_args[@]}" --time=08:00:00 --dependency=afterok:$train_meg --array=2-3%2 submit-jobs/train_elf_semantic_scaling_adapter_ladder.sbatch)")

scaled_train_deps=$train_mri:$train_meg
x1_exact=$(job_id "$(sbatch --parsable "${gpu_args[@]}" --time=04:00:00 --dependency=afterok:$scaled_train_deps submit-jobs/eval_elf_semantic_scaling_x1_exact.sbatch)")
x1_raw=$(job_id "$(sbatch --parsable "${gpu_args[@]}" --time=04:00:00 --dependency=afterok:$scaled_train_deps --array=4-5%2 submit-jobs/eval_elf_semantic_scaling_raw_ladder.sbatch)")
x1_adapter=$(job_id "$(sbatch --parsable "${gpu_args[@]}" --time=08:00:00 --dependency=afterok:$scaled_train_deps --array=4-5%2 submit-jobs/train_elf_semantic_scaling_adapter_ladder.sbatch)")

all_jobs=$x1_exact:$x1_raw:$x1_adapter:$raw_mri:$raw_meg:$adapter_mri:$adapter_meg
report=$(job_id "$(sbatch --parsable --clusters=htc --partition=short --qos=standard --time=00:15:00 --dependency=afterany:$all_jobs submit-jobs/report_elf_semantic_scaling_ladder.sbatch)")

printf 'train_mri=%s\n' "$train_mri"
printf 'train_meg=%s\n' "$train_meg"
printf 'raw_mri=%s\n' "$raw_mri"
printf 'raw_meg=%s\n' "$raw_meg"
printf 'adapter_mri=%s\n' "$adapter_mri"
printf 'adapter_meg=%s\n' "$adapter_meg"
printf 'x1_exact=%s\n' "$x1_exact"
printf 'x1_raw=%s\n' "$x1_raw"
printf 'x1_adapter=%s\n' "$x1_adapter"
printf 'report=%s\n' "$report"
