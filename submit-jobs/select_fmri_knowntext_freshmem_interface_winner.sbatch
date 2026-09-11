#!/usr/bin/env bash
#SBATCH --job-name=fmri_mem_pick
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=submit-jobs/select_fmri_knowntext_freshmem_interface_winner-%j.out
#SBATCH --error=submit-jobs/select_fmri_knowntext_freshmem_interface_winner-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
snapshot_tag=${ORACLE_SNAPSHOT_TAG:?Set ORACLE_SNAPSHOT_TAG}
run_root=/data/engs-pnpl/glandau/elf-runs
selection_dir=$run_root/fmri_knowntext_freshmem_${snapshot_tag}_selection_20260901
mkdir -p "$selection_dir"

selection_json=$selection_dir/selection.json
python - "$run_root" "$snapshot_tag" "$selection_json" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
snapshot_tag = sys.argv[2]
output = Path(sys.argv[3])
variants = [
    "frozenelf_adapter_lr1e4",
    "frozenelf_adapter_lr1e5",
    "frozenelf_lora8_attnmlp_adapter_lr1e5",
    "last4_adapter_lr5e6",
    "full_elf_adapter_lr1e6",
]
rows = []
missing = []
for variant in variants:
    tag = f"fmri_knowntext_freshmem_{snapshot_tag}_{variant}_val266_seed49_20260901"
    run_dir = run_root / tag
    metrics_path = run_dir / "best_metrics.json"
    checkpoint = run_dir / "best.pt"
    if not metrics_path.exists() or not checkpoint.exists():
        missing.append(tag)
        continue
    metrics = json.loads(metrics_path.read_text())
    quality = metrics.get("generation_quality") or {}
    retrieval = metrics.get("generation_t5_retrieval") or {}
    row = {
        "variant": variant,
        "run_tag": tag,
        "checkpoint": str(checkpoint),
        "step": metrics.get("step"),
        "wer": float(quality.get("word_error_rate", float("inf"))),
        "word_f1": float(quality.get("words_overlap", 0.0)),
        "content_f1": float(quality.get("content_words_overlap", 0.0)),
        "top1": float(retrieval.get("top1", 0.0)),
        "top5": float(retrieval.get("top5", 0.0)),
    }
    rows.append(row)
if not rows:
    raise SystemExit("No completed interface checkpoint is available")

# Primary objective matches the user's current request and the T5-prefix
# comparison: minimize validation WER, then prefer lexical/content retention.
winner = min(rows, key=lambda r: (r["wer"], -r["word_f1"], -r["content_f1"], -r["top1"]))
payload = {
    "selection_split": "predicted val266",
    "selection_rule": "lowest WER, then word F1, content F1, generated Top-1",
    "snapshot_tag": snapshot_tag,
    "missing": missing,
    "candidates": rows,
    "winner": winner,
}
output.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

winner_checkpoint=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"]["checkpoint"])' "$selection_json")
winner_tag=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"]["variant"])' "$selection_json")

cd "$PROJECT_ROOT"
test_job=$(sbatch --parsable \
  --export=ALL,INTERFACE_CHECKPOINT="$winner_checkpoint",INTERFACE_WINNER_TAG="${snapshot_tag}_${winner_tag}" \
  submit-jobs/eval_fmri_knowntext_freshmem_interface_test107.sbatch)
printf '%s\n' "$test_job" > "$selection_dir/test_job_id.txt"
echo "Submitted locked test evaluation $test_job for $winner_tag"

test_job_id=${test_job%%;*}
report_job=$(sbatch --parsable \
  --dependency=afterok:"$test_job_id" \
  --export=ALL,ORACLE_SNAPSHOT_TAG="$snapshot_tag" \
  submit-jobs/report_fmri_knowntext_freshmem.sbatch)
printf '%s\n' "$report_job" > "$selection_dir/report_job_id.txt"
echo "Submitted report job $report_job"
