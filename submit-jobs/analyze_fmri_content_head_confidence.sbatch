#!/usr/bin/env bash
#SBATCH --job-name=fmri_lexconf
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=submit-jobs/analyze_fmri_content_head_confidence-%j.out
#SBATCH --error=submit-jobs/analyze_fmri_content_head_confidence-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${OUTPUT_JSON:?OUTPUT_JSON is required}"

cd "$project_root"
export PYTHONPATH="$project_root:$project_root/src:$project_root/scripts:${PYTHONPATH:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$data_root/conda-envs/braindiffusion-elf-torch271cu118"

python scripts/analyze_fmri_content_head_confidence.py \
  --checkpoint "$CHECKPOINT" \
  --oof-semantic-npz "$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz" \
  --text-npz "$data_root/minilm/knowntext_unseenbrain_v1/oracle_exact_minilm384_train_val.npz" \
  --output "$OUTPUT_JSON"
