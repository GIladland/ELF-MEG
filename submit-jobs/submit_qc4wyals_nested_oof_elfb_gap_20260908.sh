#!/usr/bin/env bash
# Submit the leakage-safe MEG OOF -> ELF-B gap-closure ladder.

set -euo pipefail

root=/data/engs-pnpl/glandau/BrainDiffusion/ELF
run_root=/data/engs-pnpl/glandau/elf-runs
interface=/data/engs-pnpl/glandau/elf-cache/tang_themoth_qc4wyals_ada_v2/interface
oracle=$run_root/tang_apples_exact_ada002_fresh_elfb_full_memorize_all2788_ep1000_seed7_20260901/best.pt
marker=/data/engs-pnpl/glandau/elf-cache/tang_themoth_qc4wyals_nested_oof_20260908/submitted.env

for path in \
  "$root/submit-jobs/train_qc4wyals_nested_oof_folds_20260908.sbatch" \
  "$root/submit-jobs/assemble_qc4wyals_nested_oof_20260908.sbatch" \
  "$root/submit-jobs/train_qc4wyals_nested_oof_elfb_adapters_20260908.sbatch" \
  "$root/submit-jobs/eval_qc4wyals_selected_cached_interface.sbatch" \
  "$root/submit-jobs/report_qc4wyals_nested_oof_elfb_gap_20260908.sbatch" \
  "$oracle" \
  "$interface/qc4wyals_val_exact_ada002.npz" \
  "$interface/qc4wyals_val_predicted_ada002.npz"; do
  [[ -f "$path" ]] || { echo "Missing required path: $path" >&2; exit 2; }
done

if [[ -f "$marker" && "${FORCE_RESUBMIT:-0}" != 1 ]]; then
  echo "Already submitted: $marker"
  cat "$marker"
  exit 0
fi

mkdir -p "$(dirname "$marker")"
cd "$root"

oof_raw=$(sbatch --parsable --clusters=htc submit-jobs/train_qc4wyals_nested_oof_folds_20260908.sbatch)
oof=${oof_raw%%;*}
assemble_raw=$(sbatch --parsable --clusters=htc --dependency="afterok:$oof" submit-jobs/assemble_qc4wyals_nested_oof_20260908.sbatch)
assemble=${assemble_raw%%;*}
adapter_raw=$(sbatch --parsable --clusters=htc --dependency="afterok:$assemble" submit-jobs/train_qc4wyals_nested_oof_elfb_adapters_20260908.sbatch)
adapter=${adapter_raw%%;*}

submit_eval() {
  local tag=$1
  local checkpoint=$2
  local source_npz=$3
  local adapter_kind=$4
  local mismatch=$5
  local dependency=${6:-}
  local args=(--parsable --clusters=htc --array=0-4%4)
  if [[ -n "$dependency" ]]; then
    args+=(--dependency="afterok:$dependency")
  fi
  sbatch "${args[@]}" \
    --export="ALL,EVAL_SPLIT=val,SELECTED_CHECKPOINT=$checkpoint,SELECTED_TAG=$tag,SOURCE_NPZ_OVERRIDE=$source_npz,SEMANTIC_ADAPTER_KIND=$adapter_kind,INIT_E2E_ADAPTER_MISMATCH=$mismatch" \
    submit-jobs/eval_qc4wyals_selected_cached_interface.sbatch
}

exact_eval_raw=$(submit_eval oofgap_exact "$oracle" "$interface/qc4wyals_val_exact_ada002.npz" flat error)
exact_eval=${exact_eval_raw%%;*}
raw_eval_raw=$(submit_eval oofgap_raw "$oracle" "$interface/qc4wyals_val_predicted_ada002.npz" flat error)
raw_eval=${raw_eval_raw%%;*}

oof_interface=/data/engs-pnpl/glandau/elf-cache/tang_themoth_qc4wyals_nested_oof_20260908/interface
flat_checkpoint=$run_root/qc4wyals_nested_oof5_elfb_flat_context_ep300_seed49_20260908/best.pt
residual_checkpoint=$run_root/qc4wyals_nested_oof5_elfb_residual_identity_joint_ep300_seed49_20260908/best.pt
flat_eval_raw=$(submit_eval oofgap_flat "$flat_checkpoint" "$interface/qc4wyals_val_predicted_ada002.npz" flat error "$adapter")
flat_eval=${flat_eval_raw%%;*}
residual_eval_raw=$(submit_eval oofgap_residual "$residual_checkpoint" "$interface/qc4wyals_val_predicted_ada002.npz" residual_identity direct-identity "$adapter")
residual_eval=${residual_eval_raw%%;*}

report_raw=$(sbatch --parsable --clusters=htc \
  --dependency="afterok:$assemble:$exact_eval:$raw_eval:$flat_eval:$residual_eval" \
  submit-jobs/report_qc4wyals_nested_oof_elfb_gap_20260908.sbatch)
report=${report_raw%%;*}

{
  echo "SUBMITTED_AT='$(date -Is)'"
  echo "OOF_FOLDS_JOB='$oof'"
  echo "ASSEMBLY_JOB='$assemble'"
  echo "ADAPTER_JOB='$adapter'"
  echo "EXACT_EVAL_JOB='$exact_eval'"
  echo "RAW_EVAL_JOB='$raw_eval'"
  echo "FLAT_EVAL_JOB='$flat_eval'"
  echo "RESIDUAL_EVAL_JOB='$residual_eval'"
  echo "REPORT_JOB='$report'"
  echo "OOF_INTERFACE='$oof_interface'"
  echo "REPORT_DIR='$run_root/qc4wyals_nested_oof5_elfb_gap_report_20260908'"
} > "$marker"
cat "$marker"
