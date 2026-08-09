#!/usr/bin/env bash
#SBATCH --job-name=sh_cov_gate
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=0-00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=submit-jobs/gate_sherlock_sva_coverage_suite-%j.out
#SBATCH --error=submit-jobs/gate_sherlock_sva_coverage_suite-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
SUITE_DIR=${SUITE_DIR:-$DATA_ROOT/sherlock_sva/coverage_suite_v1}
MIN_COVERAGE=${MIN_COVERAGE:-0.50}
MAX_RUNS=${MAX_RUNS:-3}
TARGET_STEPS=${TARGET_STEPS:-60000}
RANKING_TSV=$SUITE_DIR/coverage_gate_ranking.tsv
REPORT_JSON=$SUITE_DIR/coverage_gate_report.json

cd "$PROJECT_ROOT"

python - "$SUITE_DIR" "$MIN_COVERAGE" "$REPORT_JSON" "$RANKING_TSV" <<'PY'
import glob
import json
import sys
from pathlib import Path

suite = Path(sys.argv[1])
threshold = float(sys.argv[2])
report_path = Path(sys.argv[3])
ranking_path = Path(sys.argv[4])
rows = []
for path_string in glob.glob(str(suite / "*_coverage.json")):
    path = Path(path_string)
    payload = json.loads(path.read_text())
    for result in payload.get("results", []):
        score = result.get("val_to_train", {}).get("word_f1", {}).get("mean")
        if score is None:
            continue
        npz = Path(result["npz"])
        summary = json.loads(npz.with_suffix(".summary.json").read_text())
        rows.append({
            "name": result["name"],
            "coverage_word_f1": float(score),
            "coverage_word_recall": float(result["val_to_train"]["word_recall"]["mean"]),
            "coverage_cosine": float(result["val_to_train"]["cosine"]["mean"]),
            "npz": str(npz),
            "train_examples": int(summary["train_examples"]),
            "passes": float(score) >= threshold,
        })
rows.sort(key=lambda row: (row["coverage_word_f1"], row["coverage_word_recall"]), reverse=True)
report = {"threshold": threshold, "ranked_datasets": rows}
report_path.write_text(json.dumps(report, indent=2))
with ranking_path.open("w") as handle:
    for row in rows:
        if row["passes"]:
            handle.write("{coverage_word_f1:.8f}\t{train_examples}\t{name}\t{npz}\n".format(**row))
print(json.dumps(report, indent=2))
PY

submitted=0
while IFS=$'\t' read -r coverage train_examples name npz; do
  [[ -n "$name" ]] || continue
  if (( submitted >= MAX_RUNS )); then
    break
  fi
  steps_per_epoch=$(( (train_examples + 63) / 64 ))
  epochs=$(( (TARGET_STEPS + steps_per_epoch - 1) / steps_per_epoch ))
  if (( epochs < 10 )); then epochs=10; fi
  run_name=${name}_minilm_k64_covgate_epoch${epochs}_fullmask_long2d
  job_id=$(sbatch --clusters=htc \
    --export=ALL,RUN_NAME="$run_name",NPZ_PATH="$npz",EPOCHS="$epochs" \
    submit-jobs/sherlock_sva_coverage_gate_train_minilm_k64.sbatch | awk '{print $NF}')
  echo "submitted job=$job_id coverage=$coverage train_examples=$train_examples epochs=$epochs name=$name"
  submitted=$((submitted + 1))
done < "$RANKING_TSV"

echo "Coverage gate submitted $submitted training runs (threshold=$MIN_COVERAGE)."
