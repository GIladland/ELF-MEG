#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}"

job_id() {
  local submitted=$1
  printf '%s' "${submitted%%;*}"
}

smoke=$(job_id "$(sbatch --parsable submit-jobs/smoke_meg_ada_t5_prefix_20260907.sbatch)")
oracle=$(job_id "$(sbatch --parsable --dependency=afterok:$smoke submit-jobs/train_meg_ada_t5_prefix_oracle_20260907.sbatch)")
selector=$(job_id "$(sbatch --parsable --dependency=afterok:$oracle submit-jobs/select_meg_ada_t5_prefix_oracle_20260907.sbatch)")
brain=$(job_id "$(sbatch --parsable --dependency=afterok:$selector submit-jobs/train_meg_ada_t5_prefix_brain_ladder_20260907.sbatch)")
evaluation=$(job_id "$(sbatch --parsable --dependency=afterany:$brain submit-jobs/eval_meg_ada_t5_prefix_ladder_20260907.sbatch)")
summary=$(job_id "$(sbatch --parsable --dependency=afterany:$evaluation submit-jobs/summarize_meg_ada_t5_prefix_ladder_20260907.sbatch)")
permutation=$(job_id "$(sbatch --parsable --dependency=afterok:$summary submit-jobs/permutation_meg_ada_t5_prefix_winner_20260907.sbatch)")

printf 'smoke=%s\n' "$smoke"
printf 'oracle_array=%s\n' "$oracle"
printf 'oracle_selector=%s\n' "$selector"
printf 'brain_ladder=%s\n' "$brain"
printf 'metrics_array=%s\n' "$evaluation"
printf 'summary=%s\n' "$summary"
printf 'permutation=%s\n' "$permutation"
