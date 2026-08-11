#!/usr/bin/env bash
#SBATCH --job-name=sva_ada_m2s_elf
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --output=submit-jobs/sherlock_sva_meg2sem_ada002_direct_elf_62l6izf0-%j.out
#SBATCH --error=submit-jobs/sherlock_sva_meg2sem_ada002_direct_elf_62l6izf0-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
PYTHON_BIN=${PYTHON_BIN:-/data/engs-pnpl/glandau/BrainDiffusion/elf-env/bin/python}

RUN_TAG=${RUN_TAG:-sherlock_sva_meg2sem_ada002_seg6000_raw_direct_elf_62l6izf0}
TRAIN_NPZ=${TRAIN_NPZ:-$DATA_ROOT/sherlock_sva/meg2sem/sherlock1to9_sva_no_coord_decl5to18_train_ada002_meg2sem_segment6000_rawfp32.npz}
VAL_NPZ=${VAL_NPZ:-$DATA_ROOT/sherlock_sva/meg2sem/sherlock12_sva_no_coord_decl5to18_val_ada002_meg2sem_segment6000_rawfp32.npz}
ADA_ELF_CHECKPOINT=${ADA_ELF_CHECKPOINT:-/data/engs-pnpl/glandau/elf-runs/sherlock1to9_sva_no_coord_decl5to18_plus_oraclevalaug64x_to_sherlock12_sva_no_coord_decl5to18_ada002_k64_epoch300_fullmask_long2d/best.pt}
ELF_CHECKPOINT=${ELF_CHECKPOINT:-$ADA_ELF_CHECKPOINT}
SEMANTIC_PROJECTOR_CHECKPOINT=${SEMANTIC_PROJECTOR_CHECKPOINT:-$ADA_ELF_CHECKPOINT}
MEG2SEM_CHECKPOINT=${MEG2SEM_CHECKPOINT:-/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/62l6izf0/checkpoints/sherlock_sva_ada002_seg6000_raw_siglip_ctr_w5_cosloss0_lr3e4_do0_bs45_seed58_r1-epoch=33-val_loss=20.88.ckpt}
OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/$RUN_TAG}

EVAL_ONLY=${EVAL_ONLY:-1}
TRAIN_MEG2SEM=${TRAIN_MEG2SEM:-0}
TRAIN_SEMANTIC_PROJECTOR=${TRAIN_SEMANTIC_PROJECTOR:-0}
UNFREEZE_ELF=${UNFREEZE_ELF:-0}
UNFREEZE_LAST_N_BLOCKS=${UNFREEZE_LAST_N_BLOCKS:-4}
SAVE_EVAL_CHECKPOINTS=${SAVE_EVAL_CHECKPOINTS:-0}
EVAL_CHECKPOINT_TOP_K=${EVAL_CHECKPOINT_TOP_K:-3}

BATCH_SIZE=${BATCH_SIZE:-12}
EPOCHS=${EPOCHS:-20}
LR=${LR:-1e-5}
MEG2SEM_LR=${MEG2SEM_LR:-$LR}
SEMANTIC_PROJECTOR_LR=${SEMANTIC_PROJECTOR_LR:-$LR}
ELF_LR=${ELF_LR:-1e-5}
EVAL_EVERY=${EVAL_EVERY:-500}
RETRIEVAL_EVAL_EVERY=${RETRIEVAL_EVAL_EVERY:-500}
INTERFACE_MONITOR_EVERY=${INTERFACE_MONITOR_EVERY:-10}
INTERFACE_MONITOR_BATCH_SIZE=${INTERFACE_MONITOR_BATCH_SIZE:-32}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-32}
EVAL_NUM_EXAMPLES=${EVAL_NUM_EXAMPLES:-75}
RETRIEVAL_NUM_EXAMPLES=${RETRIEVAL_NUM_EXAMPLES:-75}
TARGET_ENCODE_BATCH_SIZE=${TARGET_ENCODE_BATCH_SIZE:-64}

export WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
export WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-meg2sem_ada002_direct_elf}
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
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export TMPDIR=${TMPDIR:-$DATA_ROOT/tmp}

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"
cd "$PROJECT_ROOT"

for required in "$PYTHON_BIN" "$TRAIN_NPZ" "$VAL_NPZ" "$ELF_CHECKPOINT" "$SEMANTIC_PROJECTOR_CHECKPOINT" "$MEG2SEM_CHECKPOINT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 2
  fi
done
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 127
fi

echo "Running MEG2SEM ADA002 -> ADA projector -> ELF"
echo "TRAIN_NPZ=$TRAIN_NPZ"
echo "VAL_NPZ=$VAL_NPZ"
echo "MEG2SEM_CHECKPOINT=$MEG2SEM_CHECKPOINT"
echo "ELF_CHECKPOINT=$ELF_CHECKPOINT"
echo "SEMANTIC_PROJECTOR_CHECKPOINT=$SEMANTIC_PROJECTOR_CHECKPOINT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "EVAL_ONLY=$EVAL_ONLY TRAIN_MEG2SEM=$TRAIN_MEG2SEM TRAIN_SEMANTIC_PROJECTOR=$TRAIN_SEMANTIC_PROJECTOR UNFREEZE_ELF=$UNFREEZE_ELF"

args=(
  scripts/train_npz_meg_to_frozen_elf.py
  --train-npz "$TRAIN_NPZ"
  --val-npz "$VAL_NPZ"
  --output_dir "$OUTPUT_DIR"
  --checkpoint_path "$SEMANTIC_PROJECTOR_CHECKPOINT"
  --elf-checkpoint-path "$ELF_CHECKPOINT"
  --semantic-projector-checkpoint "$SEMANTIC_PROJECTOR_CHECKPOINT"
  --adapter-kind meg2sem_projector
  --meg2sem-checkpoint "$MEG2SEM_CHECKPOINT"
  --meg2sem-output-normalization auto
  --meg-standardization none
  --meg-clip-boundary -1
  --device cuda
  --num-examples 0
  --eval-num-examples "$EVAL_NUM_EXAMPLES"
  --retrieval-num-examples "$RETRIEVAL_NUM_EXAMPLES"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --eval_every "$EVAL_EVERY"
  --retrieval-eval-every "$RETRIEVAL_EVAL_EVERY"
  --retrieval-batch-size 128
  --target-encode-batch-size "$TARGET_ENCODE_BATCH_SIZE"
  --num_sampling_steps "$NUM_SAMPLING_STEPS"
  --train-timestep-mode sampling_schedule
  --train-timestep-steps 32
  --context_length 64
  --semantic-hidden-dim 4096
  --teacher-context-loss-weight 0.0
  --lr "$LR"
  --meg2sem-lr "$MEG2SEM_LR"
  --semantic-projector-lr "$SEMANTIC_PROJECTOR_LR"
  --weight_decay 1e-2
  --cfg_scale 1.0
  --self_cond_cfg_scale 1.0
  --cond_dropout_prob 0.0
  --denoiser_loss_weight 1.0
  --decoder_loss_weight 1.0
  --train-target-mask-mode full
  --interface-monitor-every "$INTERFACE_MONITOR_EVERY"
  --interface-monitor-batch-size "$INTERFACE_MONITOR_BATCH_SIZE"
  --generation-t5-retrieval
  --seed 7
  --use_wandb
  --wandb_project "$WANDB_PROJECT"
  --wandb_entity "$WANDB_ENTITY"
  --wandb_group "$WANDB_RUN_GROUP"
  --wandb_run_name "$WANDB_RUN_NAME"
  --wandb_sample_examples 16
  --wandb_notes "Direct bridge: MEG2SEM run 62l6izf0 predicts 1536-d text-embedding-ada-002 vectors from segment6000 raw SVA MEG, then routes through the ADA oracle semantic projector and ELF checkpoint. First pass is eval-only by default on packed Sherlock12 SVA val."
)

if [[ "$EVAL_ONLY" == "1" ]]; then
  args+=(--eval-only)
fi
if [[ "$TRAIN_MEG2SEM" == "1" ]]; then
  args+=(--train-meg2sem)
fi
if [[ "$TRAIN_SEMANTIC_PROJECTOR" == "1" ]]; then
  args+=(--train-semantic-projector)
fi
if [[ "$UNFREEZE_ELF" == "1" ]]; then
  args+=(--unfreeze-elf --unfreeze-last-n-blocks "$UNFREEZE_LAST_N_BLOCKS" --elf-lr "$ELF_LR")
fi
if [[ "$SAVE_EVAL_CHECKPOINTS" == "1" ]]; then
  args+=(--save-eval-checkpoints --eval-checkpoint-top-k "$EVAL_CHECKPOINT_TOP_K")
fi

"$PYTHON_BIN" "${args[@]}"
