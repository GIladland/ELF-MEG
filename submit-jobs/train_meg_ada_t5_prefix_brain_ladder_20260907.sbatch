#!/usr/bin/env bash
#SBATCH --job-name=meg_t5_brain
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=72G
#SBATCH --array=0-8%3
#SBATCH --output=submit-jobs/train_meg_ada_t5_prefix_brain_ladder_20260907-%A_%a.out
#SBATCH --error=submit-jobs/train_meg_ada_t5_prefix_brain_ladder_20260907-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
pack=$data_root/tang_themoth_qc4wyals_ada_v2/interface/qc4wyals_predicted_train_valtail_ada002.npz
root=${ROOT_OVERRIDE:-/data/engs-pnpl/glandau/elf-runs/meg_ada_t5_prefix_20260907}
oracle=$root/selected_oracle/oracle_best.pt

brain_epochs=120
eval_every=10
brain_lr=3e-5
warmup_epochs=20
warmup_lr=1e-4
hidden=1024
content_ce=1
pairing=0
lora_rank=0
lora_last=2
projector_lr=3e-6
extra=()

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    tag=raw_frozen_oracle
    brain_epochs=1
    eval_every=1
    brain_lr=0
    warmup_epochs=0
    hidden=0
    ;;
  1) tag=adapter_h1024 ;;
  2)
    tag=adapter_h2048_content_pair
    hidden=2048
    content_ce=2
    pairing=0.1
    ;;
  3)
    tag=adapter_h1024_lora4
    lora_rank=4
    pairing=0.1
    ;;
  4)
    tag=adapter_h1024_lora8
    lora_rank=8
    lora_last=4
    pairing=0.1
    ;;
  5)
    tag=joint_projector_adapter
    extra+=(--brain-unfreeze-projector)
    ;;
  6)
    tag=joint_projector_adapter_lora4_content_pair
    lora_rank=4
    content_ce=2
    pairing=0.1
    extra+=(--brain-unfreeze-projector)
    ;;
  7)
    tag=direct_projector
    hidden=0
    warmup_epochs=0
    ;;
  8)
    tag=lora4_only
    hidden=0
    warmup_epochs=0
    brain_lr=0
    lora_rank=4
    pairing=0.1
    ;;
  *) exit 2 ;;
esac

if [[ ! -s "$oracle" ]]; then
  echo "Missing selected oracle checkpoint: $oracle" >&2
  exit 3
fi

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
  --oracle-init-checkpoint "$oracle" \
  --output-dir "$root/$tag" \
  --model-name t5-small \
  --train-count 2652 \
  --val-count 110 \
  --prefix-length 16 \
  --projector-hidden-dim 1024 \
  --dropout 0.10 \
  --oracle-epochs 0 \
  --brain-epochs "$brain_epochs" \
  --brain-semantic-warmup-epochs "$warmup_epochs" \
  --eval-every "$eval_every" \
  --batch-size 64 \
  --eval-batch-size 55 \
  --brain-lr "$brain_lr" \
  --brain-semantic-warmup-lr "$warmup_lr" \
  --brain-projector-lr "$projector_lr" \
  --brain-calibrator-hidden-dim "$hidden" \
  --brain-calibrator-kind identity_residual \
  --brain-semantic-alignment-weight 1.0 \
  --brain-semantic-contrastive-weight 0.1 \
  --oracle-preservation-weight 0.5 \
  --prefix-alignment-weight 0.1 \
  --content-token-ce-weight "$content_ce" \
  --brain-pairing-margin-weight "$pairing" \
  --t5-lora-rank "$lora_rank" \
  --t5-lora-alpha 8 \
  --t5-lora-last-n-blocks "$lora_last" \
  --t5-lora-lr 1e-5 \
  --max-output-words 10 \
  --num-beams 2 \
  --derangements 3 \
  --selection-objective word_content_mean \
  --modality meg \
  --protected-split-name test26 \
  --seed 49 \
  --device cuda \
  "${extra[@]}"
