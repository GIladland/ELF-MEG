#!/usr/bin/env bash
#SBATCH --job-name=meg_t5lexsel
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=submit-jobs/rerank_meg_t5_elf_lexical_probe_20260912-%j.out
#SBATCH --error=submit-jobs/rerank_meg_t5_elf_lexical_probe_20260912-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
runs_root=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
interface=$data_root/tang_themoth_qc4wyals_nested_oof_20260908/interface
oof_input=$interface/qc4wyals_oof_predicted_train_valtail_ada002.npz
candidate_dir=$project_root/meg_t5_elf_gap_closure_20260912
output_dir=$runs_root/meg_t5_elf_lexical_probe_rerank_val110_20260912

for required in "$python_bin" "$oof_input" \
  "$candidate_dir/t5_direct_projector_metriccap10.json" \
  "$candidate_dir/lexical_composite_metriccap10.json" \
  "$candidate_dir/content_specialist_metriccap10.json" \
  "$candidate_dir/retrieval_specialist_metriccap10.json"; do
  [[ -f "$required" ]] || { echo "Missing $required" >&2; exit 2; }
done
mkdir -p "$output_dir"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd "$project_root"
exec "$python_bin" scripts/rerank_generated_candidates_by_lexical_probe.py \
  --semantic-npz "$oof_input" \
  --train-split train \
  --validation-split val \
  --metrics "$candidate_dir/t5_direct_projector_metriccap10.json" \
  --metrics "$candidate_dir/lexical_composite_metriccap10.json" \
  --metrics "$candidate_dir/content_specialist_metriccap10.json" \
  --metrics "$candidate_dir/retrieval_specialist_metriccap10.json" \
  --min-frequency 5 \
  --max-vocabulary 5000 \
  --output "$output_dir/best_metrics.json"
