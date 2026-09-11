#!/usr/bin/env bash
# Submit the full five-fold ELF-B ladder after a manually audited assembly.

set -euo pipefail

root=/data/engs-pnpl/glandau/BrainDiffusion/ELF
run_root=/data/engs-pnpl/glandau/elf-runs
oof_interface=/data/engs-pnpl/glandau/elf-cache/tang_themoth_qc4wyals_nested_oof_20260908/interface
deployment_interface=/data/engs-pnpl/glandau/elf-cache/tang_themoth_qc4wyals_ada_v2/interface
marker=/data/engs-pnpl/glandau/elf-cache/tang_themoth_qc4wyals_nested_oof_20260908/postassembly_submitted.env

for path in \
  "$root/submit-jobs/train_qc4wyals_nested_oof_elfb_adapters_20260908.sbatch" \
  "$root/submit-jobs/eval_qc4wyals_selected_cached_interface.sbatch" \
  "$root/submit-jobs/report_qc4wyals_nested_oof_elfb_gap_20260908.sbatch" \
  "$oof_interface/qc4wyals_oof_predicted_train_valtail_ada002.npz" \
  "$oof_interface/qc4wyals_oof_exact_alignment_train_valtail_ada002.npz" \
  "$oof_interface/qc4wyals_nested_oof_summary.json"; do
  [[ -f "$path" ]] || { echo "Missing required path: $path" >&2; exit 2; }
done

if [[ -f "$marker" && "${FORCE_RESUBMIT:-0}" != 1 ]]; then
  echo "Already submitted: $marker"
  cat "$marker"
  exit 0
fi

cd "$root"
adapter_raw=$(sbatch --parsable --clusters=htc --array=0-1%2 \
  submit-jobs/train_qc4wyals_nested_oof_elfb_adapters_20260908.sbatch)
adapter=${adapter_raw%%;*}

submit_eval() {
  local tag=$1
  local checkpoint=$2
  local adapter_kind=$3
  local mismatch=$4
  sbatch --parsable --clusters=htc --array=0-4%4 --dependency="afterok:$adapter" \
    --export="ALL,EVAL_SPLIT=val,SELECTED_CHECKPOINT=$checkpoint,SELECTED_TAG=$tag,SOURCE_NPZ_OVERRIDE=$deployment_interface/qc4wyals_val_predicted_ada002.npz,SEMANTIC_ADAPTER_KIND=$adapter_kind,INIT_E2E_ADAPTER_MISMATCH=$mismatch" \
    submit-jobs/eval_qc4wyals_selected_cached_interface.sbatch
}

flat_checkpoint=$run_root/qc4wyals_nested_oof5_elfb_flat_context_ep300_seed49_20260908/best.pt
residual_checkpoint=$run_root/qc4wyals_nested_oof5_elfb_residual_identity_joint_ep300_seed49_20260908/best.pt
flat_eval_raw=$(submit_eval oofgap_flat "$flat_checkpoint" flat error)
flat_eval=${flat_eval_raw%%;*}
residual_eval_raw=$(submit_eval oofgap_residual "$residual_checkpoint" residual_identity direct-identity)
residual_eval=${residual_eval_raw%%;*}
report_raw=$(sbatch --parsable --clusters=htc \
  --dependency="afterok:$flat_eval:$residual_eval" \
  submit-jobs/report_qc4wyals_nested_oof_elfb_gap_20260908.sbatch)
report=${report_raw%%;*}

{
  echo "SUBMITTED_AT='$(date -Is)'"
  echo "ASSEMBLY_MODE='manual_audited_full_2652'"
  echo "ADAPTER_JOB='$adapter'"
  echo "FLAT_EVAL_JOB='$flat_eval'"
  echo "RESIDUAL_EVAL_JOB='$residual_eval'"
  echo "REPORT_JOB='$report'"
  echo "REPORT_DIR='$run_root/qc4wyals_nested_oof5_elfb_gap_report_20260908'"
} > "$marker"
cat "$marker"
