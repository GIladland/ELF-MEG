#!/usr/bin/env bash
#SBATCH --job-name=fmri_pres_pick
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=submit-jobs/select_fmri_freshmem_semantic_preservation-%j.out
#SBATCH --error=submit-jobs/select_fmri_freshmem_semantic_preservation-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
snapshot=${ORACLE_SNAPSHOT:?Set ORACLE_SNAPSHOT}
snapshot_tag=${ORACLE_SNAPSHOT_TAG:-step024000}
run_root=/data/engs-pnpl/glandau/elf-runs
selection_dir=$run_root/fmri_freshmem_semantic_preservation_${snapshot_tag}_selection_20260901
selection_json=$selection_dir/selection.json

cd "$PROJECT_ROOT"
python scripts/select_fmri_semantic_preservation.py \
  --run-root "$run_root" \
  --snapshot-tag "$snapshot_tag" \
  --output "$selection_json"

promoted=$(python -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["promoted"]))' "$selection_json")
if [[ "$promoted" != "1" ]]; then
  echo "No validation checkpoint passed the retrieval-preservation gate; test107 remains unopened."
  report_job=$(sbatch --parsable \
    --export=ALL,ORACLE_SNAPSHOT_TAG="$snapshot_tag" \
    submit-jobs/report_fmri_freshmem_semantic_preservation.sbatch)
  printf '%s\n' "$report_job" > "$selection_dir/report_job_id.txt"
  exit 0
fi

winner_checkpoint=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"]["checkpoint"])' "$selection_json")
winner_variant=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"]["variant"])' "$selection_json")

test_job=$(sbatch --parsable \
  --export=ALL,ORACLE_SNAPSHOT="$snapshot",ORACLE_SNAPSHOT_TAG="$snapshot_tag",WINNER_CHECKPOINT="$winner_checkpoint",WINNER_VARIANT="$winner_variant" \
  submit-jobs/eval_fmri_freshmem_semantic_preservation_test107.sbatch)
printf '%s\n' "$test_job" > "$selection_dir/test_job_id.txt"
test_job_id=${test_job%%;*}

report_job=$(sbatch --parsable \
  --dependency=afterok:"$test_job_id" \
  --export=ALL,ORACLE_SNAPSHOT_TAG="$snapshot_tag" \
  submit-jobs/report_fmri_freshmem_semantic_preservation.sbatch)
printf '%s\n' "$report_job" > "$selection_dir/report_job_id.txt"

echo "Selected $winner_variant; submitted test $test_job and report $report_job"
