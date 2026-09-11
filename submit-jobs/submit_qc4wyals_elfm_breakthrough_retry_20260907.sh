#!/usr/bin/env bash
set -euo pipefail

# Retry only the failed scientific stages. The interpolation analysis and the
# upstream x16 ELF-M/qc4wyals checkpoints are already complete.
cd "${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}"

job_id() {
  local submitted=$1
  printf '%s' "${submitted%%;*}"
}

context=$(job_id "$(sbatch --parsable submit-jobs/train_qc4wyals_elfm_context_distill_20260906.sbatch)")
robust_context=$(job_id "$(sbatch --parsable submit-jobs/train_qc4wyals_elfm_context_distill_robustinit_20260906.sbatch)")
selector=$(job_id "$(sbatch --parsable --dependency=afterany:$context:$robust_context submit-jobs/select_qc4wyals_elfm_context_distill_20260906.sbatch)")
post_context=$(job_id "$(sbatch --parsable --dependency=afterok:$selector submit-jobs/train_qc4wyals_elfm_postdistill_matrix_20260906.sbatch)")

temporal_context=$(job_id "$(sbatch --parsable submit-jobs/train_qc4wyals_elfm_temporal_context_distill_20260906.sbatch)")
temporal_post=$(job_id "$(sbatch --parsable --dependency=afterok:$temporal_context submit-jobs/train_qc4wyals_elfm_temporal_postdistill_20260906.sbatch)")

multiseed=$(job_id "$(sbatch --parsable --dependency=afterany:$post_context:$temporal_post submit-jobs/eval_qc4wyals_elfm_breakthrough_multiseed_20260907.sbatch)")
summary=$(job_id "$(sbatch --parsable --dependency=afterany:$multiseed submit-jobs/summarize_qc4wyals_elfm_breakthrough_20260907.sbatch)")

printf 'context_distillation=%s\n' "$context"
printf 'robust_context_distillation=%s\n' "$robust_context"
printf 'context_selector=%s\n' "$selector"
printf 'post_context_matrix=%s\n' "$post_context"
printf 'temporal_context_distillation=%s\n' "$temporal_context"
printf 'temporal_post_context=%s\n' "$temporal_post"
printf 'multiseed_eval=%s\n' "$multiseed"
printf 'summary=%s\n' "$summary"
