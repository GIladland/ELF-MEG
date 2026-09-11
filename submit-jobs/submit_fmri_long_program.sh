#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
cd "$PROJECT_ROOT"

bridge_job=$(sbatch --parsable submit-jobs/fmri_minilm_long_bridge_val.sbatch)
e2e_job=$(sbatch --parsable submit-jobs/fmri_minilm_rawbrain_e2e_projector_val.sbatch)
ada_prep_job=$(sbatch --parsable submit-jobs/prepare_fmri_knowntext_ada002.sbatch)
bridge_job=${bridge_job%%;*}
e2e_job=${e2e_job%%;*}
ada_prep_job=${ada_prep_job%%;*}
oracle_grid_job=$(
  sbatch --parsable \
    --dependency="afterok:${ada_prep_job}" \
    submit-jobs/fmri_knowntext_oracle_scale_semantic.sbatch
)
oracle_grid_job=${oracle_grid_job%%;*}
scale_e2e_job=$(
  sbatch --parsable \
    --dependency="afterok:${oracle_grid_job}" \
    submit-jobs/fmri_rawbrain_e2e_scale_semantic_val.sbatch
)
scale_e2e_job=${scale_e2e_job%%;*}

printf 'bridge_long_array=%s\n' "$bridge_job"
printf 'rawbrain_minilm_elfb_array=%s\n' "$e2e_job"
printf 'ada_prepare=%s\n' "$ada_prep_job"
printf 'oracle_semantic_scale_array=%s dependency=afterok:%s\n' "$oracle_grid_job" "$ada_prep_job"
printf 'rawbrain_semantic_scale_array=%s dependency=afterok:%s\n' "$scale_e2e_job" "$oracle_grid_job"
