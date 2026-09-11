#!/usr/bin/env bash
#SBATCH --job-name=meg_t5_perm
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/permutation_meg_ada_t5_prefix_winner_20260907-%j.out
#SBATCH --error=submit-jobs/permutation_meg_ada_t5_prefix_winner_20260907-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
root=${ROOT_OVERRIDE:-/data/engs-pnpl/glandau/elf-runs/meg_ada_t5_prefix_20260907}

export HF_HOME=${HF_HOME:-$data_root/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export PYTHONNOUSERSITE=1
export PYTHONPATH=$data_root/python-packages/bert-score-0.3.13${PYTHONPATH:+:$PYTHONPATH}
cd "$project_root"

exec "$python_bin" scripts/run_meg_t5_prefix_winner_audit.py \
  --summary "$root/MEG_ADA_T5_PREFIX_LADDER_20260907.json" \
  --output-dir "$root"
