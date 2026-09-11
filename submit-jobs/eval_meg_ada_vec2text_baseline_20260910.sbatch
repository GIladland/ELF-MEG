#!/usr/bin/env bash
#SBATCH --job-name=meg_vec2text
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-1%2
#SBATCH --output=submit-jobs/eval_meg_ada_vec2text_baseline_20260910-%A_%a.out
#SBATCH --error=submit-jobs/eval_meg_ada_vec2text_baseline_20260910-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
interface=$data_root/tang_themoth_qc4wyals_ada_v2/interface
output_root=${OUTPUT_ROOT:-/data/engs-pnpl/glandau/elf-runs/meg_ada_vec2text_baseline_20260910}
key_file=${OPENAI_API_KEY_FILE:-/data/engs-pnpl/glandau/BrainDiffusion/secrets/openai_api_key}

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    label=predicted_qc4wyals
    input_npz=$interface/qc4wyals_val_predicted_ada002.npz
    ;;
  1)
    label=exact_ada_ceiling
    input_npz=$interface/qc4wyals_val_exact_ada002.npz
    ;;
  *) exit 2 ;;
esac

for required in "$input_npz" "$key_file"; do
  [[ -s "$required" ]] || { echo "Missing $required" >&2; exit 2; }
done

export XDG_CACHE_HOME=$data_root/xdg-cache
# The ADA Vec2Text hypothesizer/corrector and T5-base are already cached here.
export HF_HOME=$data_root/huggingface
export HF_HUB_CACHE=/home/glandau/.cache/huggingface/hub
export HF_DATASETS_CACHE=$data_root/huggingface/datasets
export TRANSFORMERS_CACHE=$data_root/huggingface/transformers
export TORCH_HOME=$data_root/torch
export PIP_CACHE_DIR=$data_root/pip-cache
export CONDA_PKGS_DIRS=$data_root/conda-pkgs
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OPENAI_API_KEY
OPENAI_API_KEY="$(tr -d '\r\n' < "$key_file")"

runtime_root=${SLURM_TMPDIR:-/tmp}/${USER}-vec2text-runtime
mkdir -p "$output_root/$label" "$data_root/tmp" "$runtime_root"
export TMPDIR=${SLURM_TMPDIR:-$data_root/tmp}
cd "$project_root"

# Current RedHat9 GPU images do not expose the old cluster-wide `conda`
# command. Reuse the persistent CUDA environment and install Vec2Text's small
# missing runtime dependencies on node-local scratch.
"$python_bin" -m pip install --no-deps --target "$runtime_root" \
  "vec2text==0.0.13" \
  "evaluate==0.4.3" \
  "accelerate==0.33.0" \
  "openai==2.16.0" \
  "sentence-transformers==3.0.1" \
  "tenacity==9.1.2" \
  "scikit-learn==1.5.2" \
  "threadpoolctl==3.5.0" \
  "psutil==6.0.0" \
  "distro==1.9.0" \
  "jiter==0.12.0" \
  "sniffio==1.3.1"
export PYTHONPATH=$runtime_root${PYTHONPATH:+:$PYTHONPATH}

raw_json=$output_root/$label/vec2text_steps20_beam4_wordcap10.json
"$python_bin" scripts/decode_vec2text_npz.py \
  --input-npz "$input_npz" \
  --output-json "$raw_json" \
  --embedder text-embedding-ada-002 \
  --num-steps "${VEC2TEXT_STEPS:-20}" \
  --sequence-beam-width "${VEC2TEXT_BEAM_WIDTH:-4}" \
  --batch-size "${VEC2TEXT_BATCH_SIZE:-8}" \
  --word-cap 10 \
  --seed 49 \
  --device cuda

# Add the same T5-small 110-way generated-text retrieval used by ELF.
for variant in "$raw_json" "${raw_json%.json}.native.json"; do
  "$python_bin" scripts/add_t5_retrieval_to_metrics.py \
    --metrics-json "$variant" \
    --output "${variant%.json}.t5retrieval.json" \
    --device cuda \
    --local-files-only
done
