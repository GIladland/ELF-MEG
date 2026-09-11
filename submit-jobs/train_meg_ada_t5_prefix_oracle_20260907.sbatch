#!/usr/bin/env bash
#SBATCH --job-name=meg_t5_oracle
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --array=0-2%3
#SBATCH --output=submit-jobs/train_meg_ada_t5_prefix_oracle_20260907-%A_%a.out
#SBATCH --error=submit-jobs/train_meg_ada_t5_prefix_oracle_20260907-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
pack=$data_root/tang_themoth_qc4wyals_ada_v2/interface/qc4wyals_predicted_train_valtail_ada002.npz
root=/data/engs-pnpl/glandau/elf-runs/meg_ada_t5_prefix_20260907

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) tag=lr1e4; oracle_lr=1e-4 ;;
  1) tag=lr3e4; oracle_lr=3e-4 ;;
  2) tag=lr6e4; oracle_lr=6e-4 ;;
  *) exit 2 ;;
esac

export HF_HOME=${HF_HOME:-$data_root/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export TMPDIR=${TMPDIR:-$data_root/tmp}
mkdir -p "$TMPDIR" "$root"
cd "$project_root"

exec "$python_bin" scripts/train_fmri_minilm_t5_prefix.py \
  --precomputed-semantic-npz "$pack" \
  --output-dir "$root/oracle_$tag" \
  --model-name t5-small \
  --train-count 2652 \
  --val-count 110 \
  --prefix-length 16 \
  --projector-hidden-dim 1024 \
  --dropout 0.10 \
  --oracle-epochs 100 \
  --brain-epochs 0 \
  --eval-every 5 \
  --batch-size 64 \
  --eval-batch-size 55 \
  --oracle-lr "$oracle_lr" \
  --max-output-words 10 \
  --num-beams 2 \
  --derangements 0 \
  --selection-objective word_content_mean \
  --modality meg \
  --protected-split-name test26 \
  --seed 49 \
  --device cuda
