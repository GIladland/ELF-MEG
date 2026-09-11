#!/usr/bin/env bash
#SBATCH --job-name=meg_t5_smoke
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=submit-jobs/smoke_meg_ada_t5_prefix_20260907-%j.out
#SBATCH --error=submit-jobs/smoke_meg_ada_t5_prefix_20260907-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
pack=$data_root/tang_themoth_qc4wyals_ada_v2/interface/qc4wyals_predicted_train_valtail_ada002.npz
root=/data/engs-pnpl/glandau/elf-runs/meg_ada_t5_prefix_20260907/smoke_contract

export HF_HOME=${HF_HOME:-$data_root/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
cd "$project_root"

"$python_bin" tests/test_fmri_t5_prefix_smoke.py

common=(
  --precomputed-semantic-npz "$pack"
  --model-name t5-small
  --train-count 2652
  --val-count 110
  --prefix-length 16
  --projector-hidden-dim 1024
  --dropout 0.10
  --eval-every 1
  --batch-size 64
  --eval-batch-size 55
  --max-output-words 10
  --num-beams 1
  --selection-objective word_content_mean
  --modality meg
  --protected-split-name test26
  --seed 49
  --device cuda
)

"$python_bin" scripts/train_fmri_minilm_t5_prefix.py \
  "${common[@]}" \
  --output-dir "$root/oracle" \
  --oracle-epochs 1 \
  --brain-epochs 0 \
  --derangements 0

"$python_bin" scripts/train_fmri_minilm_t5_prefix.py \
  "${common[@]}" \
  --oracle-init-checkpoint "$root/oracle/oracle_best.pt" \
  --output-dir "$root/raw" \
  --oracle-epochs 0 \
  --brain-epochs 1 \
  --brain-lr 0 \
  --brain-calibrator-hidden-dim 0 \
  --brain-calibrator-kind identity_residual \
  --derangements 1

test -s "$root/oracle/oracle_best_metrics.json"
test -s "$root/raw/best_metrics.json"
