#!/usr/bin/env bash
#SBATCH --job-name=elf_scale_raw_cpu
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --array=0,3-5%4
#SBATCH --output=submit-jobs/eval_elf_semantic_scaling_raw_ladder_cpu-%A_%a.out
#SBATCH --error=submit-jobs/eval_elf_semantic_scaling_raw_ladder_cpu-%A_%a.err

set -euo pipefail
export BERT_DEVICE=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
bash "$project_root/submit-jobs/eval_elf_semantic_scaling_raw_ladder.sbatch"
