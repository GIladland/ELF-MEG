#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
cd "$project_root"

job_id() {
  local value=$1
  printf '%s\n' "${value%%;*}"
}

prep=$(job_id "$(sbatch --parsable submit-jobs/prepare_fmri_minilm_knowntext_oof.sbatch)")
oracle=$(job_id "$(sbatch --parsable --dependency=afterok:$prep submit-jobs/fmri_minilm_knowntext_oracle_all12098.sbatch)")
mapper=$(job_id "$(sbatch --parsable --dependency=afterok:$prep submit-jobs/train_fmri_minilm_knowntext_oof_mapper.sbatch)")
adapter=$(job_id "$(sbatch --parsable --dependency=afterok:$prep:$oracle submit-jobs/fmri_minilm_knowntext_oof_adapter.sbatch)")
e2e=$(job_id "$(sbatch --parsable --dependency=afterok:$adapter submit-jobs/fmri_minilm_knowntext_oof_e2e.sbatch)")
final_eval=$(job_id "$(sbatch --parsable --dependency=afterok:$oracle:$mapper:$adapter:$e2e submit-jobs/eval_fmri_minilm_knowntext_unseenbrain.sbatch)")

echo "prep=$prep"
echo "oracle_all_sentences=$oracle"
echo "oof_mapper_array=$mapper"
echo "oof_adapter=$adapter"
echo "oof_full_e2e=$e2e"
echo "final_eval_array=$final_eval"
