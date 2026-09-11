#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}"
root=/data/engs-pnpl/glandau/elf-runs/meg_ada_t5_prefix_x16_knowntext_20260908

job_id() {
  local submitted=$1
  printf '%s' "${submitted%%;*}"
}

oracle=$(job_id "$(sbatch --parsable --export=ALL,ROOT_OVERRIDE=$root submit-jobs/train_meg_ada_t5_prefix_x16_knowntext_oracle_20260908.sbatch)")
selector=$(job_id "$(sbatch --parsable --dependency=afterok:$oracle --export=ALL,ROOT_OVERRIDE=$root submit-jobs/select_meg_ada_t5_prefix_oracle_20260907.sbatch)")
brain=$(job_id "$(sbatch --parsable --dependency=afterok:$selector --export=ALL,ROOT_OVERRIDE=$root submit-jobs/train_meg_ada_t5_prefix_brain_ladder_20260907.sbatch)")
evaluation=$(job_id "$(sbatch --parsable --dependency=afterany:$brain --export=ALL,ROOT_OVERRIDE=$root submit-jobs/eval_meg_ada_t5_prefix_ladder_20260907.sbatch)")
summary=$(job_id "$(sbatch --parsable --dependency=afterany:$evaluation --export=ALL,ROOT_OVERRIDE=$root submit-jobs/summarize_meg_ada_t5_prefix_ladder_20260907.sbatch)")
permutation=$(job_id "$(sbatch --parsable --dependency=afterok:$summary --export=ALL,ROOT_OVERRIDE=$root submit-jobs/permutation_meg_ada_t5_prefix_winner_20260907.sbatch)")

printf 'x16_oracle_array=%s\n' "$oracle"
printf 'oracle_selector=%s\n' "$selector"
printf 'brain_ladder=%s\n' "$brain"
printf 'metrics_array=%s\n' "$evaluation"
printf 'summary=%s\n' "$summary"
printf 'permutation=%s\n' "$permutation"
