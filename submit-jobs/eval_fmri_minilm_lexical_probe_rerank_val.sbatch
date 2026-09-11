#!/usr/bin/env bash
#SBATCH --job-name=fmri_lex_rerank
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=submit-jobs/eval_fmri_minilm_lexical_probe_rerank_val-%j.out
#SBATCH --error=submit-jobs/eval_fmri_minilm_lexical_probe_rerank_val-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
RUNS_ROOT=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
OUTPUT_DIR=$RUNS_ROOT/fmri_minilm_stage2_lexicalprobe_rerank234_val266_20260830

mkdir -p "$OUTPUT_DIR"
candidate_files=()
while IFS= read -r path; do
  candidate_files+=("$path")
done < <(
  find \
    "$RUNS_ROOT"/fmri_minilm_rawbrain_e2e_projector_long100_resdelaytok_h1024_l2_h8_storybal_b32_seed49_20260830 \
    "$RUNS_ROOT"/fmri_minilm_long100_*_20260830 \
    -maxdepth 1 -type f -name 'eval_step_*.json' -print | sort
)
for cfg in 0p5 1p0 1p5; do
  for seed in 7 23 49 101; do
    candidate_files+=("$RUNS_ROOT/fmri_minilm_stage2_sampler_seed${seed}_cfg${cfg}_steps32_val266_20260830/best_metrics.json")
  done
done
if (( ${#candidate_files[@]} < 200 )); then
  echo "Unexpectedly small lexical candidate pool: ${#candidate_files[@]}" >&2
  exit 1
fi

metrics_args=()
for metrics in "${candidate_files[@]}"; do
  metrics_args+=(--metrics "$metrics")
done

cd "$PROJECT_ROOT"
python scripts/rerank_generated_candidates_by_lexical_probe.py \
  --semantic-npz "$DATA_ROOT/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz" \
  "${metrics_args[@]}" \
  --min-frequency 5 \
  --max-vocabulary 5000 \
  --output "$OUTPUT_DIR/best_metrics.json"

printf '%s\n' "${candidate_files[@]}" > "$OUTPUT_DIR/candidate_files.txt"
echo "Train-only lexical-probe reranking written to $OUTPUT_DIR"
