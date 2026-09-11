#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}"

job_id() {
  local submitted=$1
  printf '%s' "${submitted%%;*}"
}

prep=$(job_id "$(sbatch --parsable submit-jobs/prepare_qc4wyals_elfm_interpolation_ladder_20260906.sbatch)")
interpolation=$(job_id "$(sbatch --parsable --dependency=afterok:$prep submit-jobs/eval_qc4wyals_elfm_interpolation_ladder_20260906.sbatch)")

context=$(job_id "$(sbatch --parsable submit-jobs/train_qc4wyals_elfm_context_distill_20260906.sbatch)")
robust_source_job=${ROBUST_ADAPTER_JOB_ID:-8732390_2}
robust_context=$(job_id "$(sbatch --parsable --dependency=afterany:$robust_source_job submit-jobs/train_qc4wyals_elfm_context_distill_robustinit_20260906.sbatch)")
selector=$(job_id "$(sbatch --parsable --dependency=afterany:$context:$robust_context submit-jobs/select_qc4wyals_elfm_context_distill_20260906.sbatch)")
post_context=$(job_id "$(sbatch --parsable --dependency=afterok:$selector submit-jobs/train_qc4wyals_elfm_postdistill_matrix_20260906.sbatch)")

temporal_context=$(job_id "$(sbatch --parsable submit-jobs/train_qc4wyals_elfm_temporal_context_distill_20260906.sbatch)")
temporal_post=$(job_id "$(sbatch --parsable --dependency=afterok:$temporal_context submit-jobs/train_qc4wyals_elfm_temporal_postdistill_20260906.sbatch)")

multiseed=$(job_id "$(sbatch --parsable --dependency=afterany:$post_context:$temporal_post submit-jobs/eval_qc4wyals_elfm_breakthrough_multiseed_20260907.sbatch)")
summary=$(job_id "$(sbatch --parsable --dependency=afterany:$multiseed submit-jobs/summarize_qc4wyals_elfm_breakthrough_20260907.sbatch)")

printf 'interpolation_prepare=%s\n' "$prep"
printf 'interpolation_eval=%s\n' "$interpolation"
printf 'context_distillation=%s\n' "$context"
printf 'robust_context_distillation=%s\n' "$robust_context"
printf 'context_selector=%s\n' "$selector"
printf 'post_context_matrix=%s\n' "$post_context"
printf 'temporal_context_distillation=%s\n' "$temporal_context"
printf 'temporal_post_context=%s\n' "$temporal_post"
printf 'multiseed_eval=%s\n' "$multiseed"
printf 'summary=%s\n' "$summary"
