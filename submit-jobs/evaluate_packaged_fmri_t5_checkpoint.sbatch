#!/usr/bin/env bash
#SBATCH --job-name=fmri_pkg_eval
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/evaluate_packaged_fmri_t5_checkpoint-%j.out
#SBATCH --error=submit-jobs/evaluate_packaged_fmri_t5_checkpoint-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=${PYTHON_BIN:-$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python}
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${SEMANTIC_NPZ:?SEMANTIC_NPZ is required}"
: "${TEXT_NPZ:?TEXT_NPZ is required}"
: "${EXPECTED_COUNT:?EXPECTED_COUNT is required}"
: "${SPLIT_NAME:?SPLIT_NAME is required}"
: "${OUTPUT_JSON:?OUTPUT_JSON is required}"

cd "$project_root"
export PYTHONPATH="$project_root:$project_root/src:$project_root/scripts:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$data_root/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}

selection_args=()
if [[ -n "${SEMANTIC_CONDITION_KEY:-}" ]]; then
  selection_args+=(--semantic-condition-key "$SEMANTIC_CONDITION_KEY")
fi
if [[ -n "${SEMANTIC_CONDITION_VALUE:-}" ]]; then
  selection_args+=(--semantic-condition-value "$SEMANTIC_CONDITION_VALUE")
fi
if [[ -n "${TEXT_TAIL_COUNT:-}" ]]; then
  selection_args+=(--text-tail-count "$TEXT_TAIL_COUNT")
fi

"$python_bin" scripts/evaluate_packaged_fmri_t5_checkpoint.py \
  --checkpoint "$CHECKPOINT" \
  --semantic-npz "$SEMANTIC_NPZ" \
  --semantic-key "${SEMANTIC_KEY:-pred}" \
  --text-npz "$TEXT_NPZ" \
  --sentence-key "${SENTENCE_KEY:-sentence}" \
  "${selection_args[@]}" \
  --expected-count "$EXPECTED_COUNT" \
  --split-name "$SPLIT_NAME" \
  --output "$OUTPUT_JSON"
