#!/usr/bin/env bash
#SBATCH --job-name=qc4_oofjoin
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/assemble_qc4wyals_nested_oof_20260908-%j.out
#SBATCH --error=submit-jobs/assemble_qc4wyals_nested_oof_20260908-%j.err

set -euo pipefail

brain_root=/data/engs-pnpl/glandau/BrainDiffusion/ELF
python_bin=/data/engs-pnpl/glandau/conda/envs/meg2sem/bin/python
source_train=/data/engs-pnpl/glandau/elf-cache/tang_themoth_apples_exact_ada002_textmid_seg10_v1/packed/tang_themoth_exact384_ada002_textmid_seg10000_train_dedup_fp16.npz
raw_root=/data/engs-pnpl/glandau/elf-cache/tang_themoth_qc4wyals_ada_v2/interface
oof_root=/data/engs-pnpl/glandau/elf-cache/tang_themoth_qc4wyals_nested_oof_20260908
interface=$oof_root/interface

fold_args=()
for fold in 0 1 2 3 4; do
  path=$oof_root/qc4wyals_oof_outer_fold${fold}_predicted_ada002.npz
  [[ -f "$path" ]] || { echo "Missing fold prediction $path" >&2; exit 2; }
  fold_args+=(--fold-prediction "$path")
done

cd "$brain_root"
exec "$python_bin" scripts/assemble_grouped_meg_oof_predictions.py \
  --source-train-npz "$source_train" \
  "${fold_args[@]}" \
  --val-prediction "$raw_root/qc4wyals_val_predicted_ada002.npz" \
  --output-dir "$interface" \
  --expected-train-rows 2652 \
  --expected-val-rows 110
