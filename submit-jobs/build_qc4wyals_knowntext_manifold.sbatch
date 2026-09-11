#!/usr/bin/env bash
#SBATCH --job-name=qc4w_manifold
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --output=submit-jobs/build_qc4wyals_knowntext_manifold-%j.out
#SBATCH --error=submit-jobs/build_qc4wyals_knowntext_manifold-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=${PYTHON_BIN:-$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python}
interface_dir=$data_root/tang_themoth_qc4wyals_ada_v2/interface
knowntext_npz=$data_root/tang_themoth_apples_exact_ada002_textmid_seg10_v1/tang_apples_ada002_knowntext_all2788_testfirst.npz
output_dir=$data_root/tang_themoth_qc4wyals_ada_v2/manifold

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}

cd "$project_root"
exec "$python_bin" scripts/build_qc4wyals_knowntext_manifold.py \
  --interface-dir "$interface_dir" \
  --knowntext-npz "$knowntext_npz" \
  --output-dir "$output_dir" \
  --max-k 64 \
  --query-batch-size 128
