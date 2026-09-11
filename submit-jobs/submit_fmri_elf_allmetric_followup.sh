#!/usr/bin/env bash
# Submit leakage-safe ELF all-metric follow-ups after focused regressions.

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
cd "$project_root"

test_job=$(
  sbatch --clusters=htc --parsable \
    submit-jobs/test_fmri_elf_content_winner.sbatch
)
test_job=${test_job%%;*}

common_margin_job=$(
  sbatch --clusters=htc --parsable \
    --dependency="afterok:${test_job}" \
    --array=22-34%4 \
    --export=ALL,CONTENT_TOPK_OVERRIDE=5,RUN_PREFIX_OVERRIDE=fmri_minilm_elf_commonmargin_top5 \
    submit-jobs/eval_fmri_elf_ordered_content_screen.sbatch
)

strict_top3_job=$(
  sbatch --clusters=htc --parsable \
    --dependency="afterok:${test_job}" \
    --array=16-21%4 \
    --export=ALL,CONTENT_TOPK_OVERRIDE=3,RUN_PREFIX_OVERRIDE=fmri_minilm_elf_reservedcap_top3 \
    submit-jobs/eval_fmri_elf_ordered_content_screen.sbatch
)

strict_sequence_job=$(
  sbatch --clusters=htc --parsable \
    --dependency="afterok:${test_job}" \
    --array=16-21%4 \
    --export=ALL,CONTENT_TOPK_OVERRIDE=5,CONTENT_MODE_OVERRIDE=sequence,RUN_PREFIX_OVERRIDE=fmri_minilm_elf_reservedcap_sequence_top5 \
    submit-jobs/eval_fmri_elf_ordered_content_screen.sbatch
)

content_weighted_head_job=$(
  sbatch --clusters=htc --parsable \
    --array=4-7%2 \
    --export=ALL,EPOCHS_OVERRIDE=60,DERANGEMENTS_OVERRIDE=5 \
    submit-jobs/probe_fmri_ordered_word_head.sbatch
)

printf 'tests=%s\n' "$test_job"
printf 'common_margin_top5=%s\n' "$common_margin_job"
printf 'reserved_cap_top3=%s\n' "$strict_top3_job"
printf 'reserved_cap_sequence_top5=%s\n' "$strict_sequence_job"
printf 'content_weighted_ordered_heads=%s\n' "$content_weighted_head_job"
printf '%s\n' 'contract=train11725/val266 only; diffusion test107 is not loaded'
