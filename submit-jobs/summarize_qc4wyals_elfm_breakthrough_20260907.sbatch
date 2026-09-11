#!/usr/bin/env bash
#SBATCH --job-name=elfm_brsum
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=submit-jobs/summarize_qc4wyals_elfm_breakthrough_20260907-%j.out
#SBATCH --error=submit-jobs/summarize_qc4wyals_elfm_breakthrough_20260907-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
root=/data/engs-pnpl/glandau/elf-runs
output=$root/meg_elfm_breakthrough_summary_20260907
mkdir -p "$output"
arms=(
  postctx_frozenelf_balanced
  postctx_lora4_balanced
  postctx_lora4_strongcontent
  postctx_frozenadapter_lora4_strongcontent
  postctx_lora8last8_strongcontent
  postctx_fullmeg_lora4_balanced
  temporal_frozenelf
  temporal_lora4last4
  temporal_lora8last8
)
for arm in "${arms[@]}"; do
  pattern="$root/eval_meg_elfm_breakthrough_${arm}_val110_seed*_20260907"
  if compgen -G "$pattern" >/dev/null; then
    "$python" "$project_root/scripts/summarize_multiseed_generation.py" "$pattern" \
      --output-json "$output/${arm}.json"
  else
    echo "No completed evaluations for $arm" >&2
  fi
done
