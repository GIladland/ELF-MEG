#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
cd "$PROJECT_ROOT"

job_id() {
  local raw=$1
  raw=${raw%%;*}
  raw=${raw%%_*}
  printf '%s\n' "$raw"
}

prep=$(job_id "$(sbatch --parsable submit-jobs/prepare_tang_themoth_apples_meg2sem.sbatch)")
train=$(job_id "$(sbatch --parsable --dependency=afterok:$prep submit-jobs/train_tang_themoth_apples_meg2sem.sbatch)")
eval_job=$(job_id "$(sbatch --parsable --dependency=afterok:$train submit-jobs/eval_tang_themoth_apples_meg2sem2text.sbatch)")

echo "Tang/TheMoth apples-to-apples chain submitted:"
echo "  prepare=$prep"
echo "  meg2sem_train=$train"
echo "  shared_elf_val_test=$eval_job"
