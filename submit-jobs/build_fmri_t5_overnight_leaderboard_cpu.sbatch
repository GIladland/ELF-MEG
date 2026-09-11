#!/usr/bin/env bash
#SBATCH --job-name=fmri_night_board
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=submit-jobs/build_fmri_t5_overnight_leaderboard_cpu-%j.out
#SBATCH --error=submit-jobs/build_fmri_t5_overnight_leaderboard_cpu-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
: "${CANDIDATE_MANIFEST:?CANDIDATE_MANIFEST is required}"
: "${OUTPUT_JSON:?OUTPUT_JSON is required}"
: "${OUTPUT_CSV:?OUTPUT_CSV is required}"

cd "$project_root"
export PYTHONPATH="$project_root:$project_root/src:$project_root/scripts:${PYTHONPATH:-}"
mapfile -t candidates < "$CANDIDATE_MANIFEST"
candidate_args=()
for candidate in "${candidates[@]}"; do
  [[ -n "$candidate" ]] && candidate_args+=(--candidate "$candidate")
done

"$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python" \
  scripts/build_fmri_t5_overnight_leaderboard.py \
  --text-npz "$data_root/minilm/knowntext_unseenbrain_v1/oracle_exact_minilm384_train_val.npz" \
  --allow-missing \
  --output-json "$OUTPUT_JSON" \
  --output-csv "$OUTPUT_CSV" \
  "${candidate_args[@]}"
