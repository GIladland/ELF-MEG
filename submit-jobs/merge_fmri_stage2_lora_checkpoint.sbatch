#!/usr/bin/env bash
#SBATCH --job-name=fmri_merge_lora
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/merge_fmri_stage2_lora_checkpoint-%j.out
#SBATCH --error=submit-jobs/merge_fmri_stage2_lora_checkpoint-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
ENV_ROOT=${ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
: "${INPUT_CHECKPOINT:?INPUT_CHECKPOINT is required}"
: "${OUTPUT_CHECKPOINT:?OUTPUT_CHECKPOINT is required}"

if ! command -v conda >/dev/null 2>&1; then
  source /apps/system/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_ROOT"
cd "$PROJECT_ROOT"
python scripts/merge_lora_checkpoint.py --input "$INPUT_CHECKPOINT" --output "$OUTPUT_CHECKPOINT"
stat -c "%n %s" "$INPUT_CHECKPOINT" "$OUTPUT_CHECKPOINT"
