#!/usr/bin/env bash
#SBATCH --job-name=sva_neuro_tctx
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --output=submit-jobs/sherlock_sva_neurobridge_teacherctx_oracle_frozen_minilm_k64_epoch100_long2d-%j.out
#SBATCH --error=submit-jobs/sherlock_sva_neurobridge_teacherctx_oracle_frozen_minilm_k64_epoch100_long2d-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}

RUN_TAG=${RUN_TAG:-sherlock_sva_neurobridge_teacherctx_oracle_frozen_minilm_k64_epoch100_long2d}
TRAIN_NPZ=${TRAIN_NPZ:-$DATA_ROOT/sherlock_sva/meg2sem/sherlock1to9_sva_no_coord_decl5to18_train_minilm_meg2sem_segment3000_fp16.npz}
ELF_CHECKPOINT=${ELF_CHECKPOINT:-/data/engs-pnpl/glandau/elf-runs/sherlock1to9_sva_no_coord_decl5to18_plus_oraclevalaug64x_to_sherlock12_sva_no_coord_decl5to18_minilm_k64_epoch300_fullmask_long2d/best.pt}
OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/$RUN_TAG}
MEG_STANDARDIZATION=${MEG_STANDARDIZATION:-none}
MEG_CLIP_BOUNDARY=${MEG_CLIP_BOUNDARY:--1}

export SKIP_ARC_SETUP=${SKIP_ARC_SETUP:-1}
export WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
export WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-neuro_bridge_frozen_elf}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$RUN_TAG}
export WANDB_DIR=${WANDB_DIR:-$DATA_ROOT/wandb}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-$DATA_ROOT/wandb-data}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$DATA_ROOT/wandb-cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$DATA_ROOT/xdg-cache}
export HF_HOME=${HF_HOME:-$DATA_ROOT/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export TORCH_HOME=${TORCH_HOME:-$DATA_ROOT/torch}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$DATA_ROOT/pip-cache}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$DATA_ROOT/conda-pkgs}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export TMPDIR=${TMPDIR:-$DATA_ROOT/tmp}

if ! command -v conda >/dev/null 2>&1; then
  for conda_sh in \
    "${CONDA_SH:-}" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/data/engs-pnpl/glandau/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -n "$conda_sh" && -f "$conda_sh" ]]; then
      source "$conda_sh"
      break
    fi
  done
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found; set CONDA_SH before launching." >&2
  exit 127
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"
cd "$PROJECT_ROOT"

ENV_STAMP_PATH="$TMP_ENV_ROOT/.elf_env_stamp"
if [[ "$SKIP_ARC_SETUP" == "1" && -d "$TMP_ENV_ROOT/bin" && -f "$ENV_STAMP_PATH" && "$(cat "$ENV_STAMP_PATH")" == "$ENV_STAMP" ]]; then
  echo "Skipping ARC env setup; reusing TMP_ENV_ROOT=$TMP_ENV_ROOT ENV_STAMP=$ENV_STAMP"
else
  echo "Preparing ARC env at $TMP_ENV_ROOT"
  PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" TMP_ENV_ROOT="$TMP_ENV_ROOT" ENV_STAMP="$ENV_STAMP" \
    bash "$PROJECT_ROOT/scripts/arc_setup_elf_env.sh"
fi
conda activate "$TMP_ENV_ROOT"

echo "Training MEG -> frozen semantic-projector context -> frozen ELF"
echo "TRAIN_NPZ=$TRAIN_NPZ"
echo "ELF_CHECKPOINT=$ELF_CHECKPOINT"
echo "OUTPUT_DIR=$OUTPUT_DIR"

python scripts/train_npz_meg_to_frozen_elf.py \
  --train-npz "$TRAIN_NPZ" \
  --output_dir "$OUTPUT_DIR" \
  --checkpoint_path "$ELF_CHECKPOINT" \
  --meg-standardization "$MEG_STANDARDIZATION" \
  --meg-clip-boundary "$MEG_CLIP_BOUNDARY" \
  --device cuda \
  --num-examples 0 \
  --val-num-examples 512 \
  --eval-num-examples 64 \
  --retrieval-num-examples 64 \
  --batch_size 64 \
  --epochs 100 \
  --eval_every 500 \
  --retrieval-eval-every 500 \
  --retrieval-batch-size 128 \
  --num_sampling_steps 32 \
  --context_length 64 \
  --semantic-hidden-dim 4096 \
  --teacher-context-loss-weight 1.0 \
  --merger-channels 306 \
  --conv-channels 320 \
  --num-conv-layers 10 \
  --adapter-dropout 0.1 \
  --norm-type batch \
  --lr 1e-4 \
  --weight_decay 1e-2 \
  --cfg_scale 1.0 \
  --self_cond_cfg_scale 1.0 \
  --train-target-mask-mode full \
  --generation-t5-retrieval \
  --seed 7 \
  --use_wandb \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_group "$WANDB_RUN_GROUP" \
  --wandb_run_name "$WANDB_RUN_NAME" \
  --wandb_sample_examples 16 \
  --wandb_notes "Teacher-context MEG bridge. Frozen oracle MiniLM K64 full-mask ELF checkpoint plus frozen semantic projector; train only MEGContextAdapter to match the projector's 64x512 context tokens from exact Sherlock1-9 simplified SVA MEG windows. Eval uses 512 held-out SVA train rows."
