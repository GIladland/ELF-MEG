#!/usr/bin/env bash
#SBATCH --job-name=sva_m2s_elf
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=submit-jobs/sherlock_sva_meg2sem_minilm_direct_elf_dt8z7o55-%j.out
#SBATCH --error=submit-jobs/sherlock_sva_meg2sem_minilm_direct_elf_dt8z7o55-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}

RUN_TAG=${RUN_TAG:-sherlock_sva_meg2sem_minilm_direct_elf_dt8z7o55}
TRAIN_NPZ=${TRAIN_NPZ:-$DATA_ROOT/sherlock_sva/meg2sem/sherlock1to9_sva_no_coord_decl5to18_train_minilm_meg2sem_segment3000_fp16.npz}
VAL_NPZ=${VAL_NPZ:-}
ELF_CHECKPOINT=${ELF_CHECKPOINT:-/data/engs-pnpl/glandau/elf-runs/sherlock1to9_sva_no_coord_decl5to18_plus_oraclevalaug64x_to_sherlock12_sva_no_coord_decl5to18_minilm_k64_epoch300_fullmask_long2d/best.pt}
SEMANTIC_PROJECTOR_CHECKPOINT=${SEMANTIC_PROJECTOR_CHECKPOINT:-$ELF_CHECKPOINT}
OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/$RUN_TAG}

MEG2SEM_RUN_ID=${MEG2SEM_RUN_ID:-dt8z7o55}
MEG2SEM_ROOT=${MEG2SEM_ROOT:-/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM}
MEG2SEM_CHECKPOINT_DIR=${MEG2SEM_CHECKPOINT_DIR:-$MEG2SEM_ROOT/wandb/$MEG2SEM_RUN_ID/checkpoints}
MEG2SEM_RECOMMENDED_CHECKPOINT=${MEG2SEM_RECOMMENDED_CHECKPOINT:-$MEG2SEM_CHECKPOINT_DIR/sherlock_sva_minilm_books1to9_to_sherlock12_bs60_seed49-epoch=852-val_loss=26.45.ckpt}
MEG2SEM_SELECTION_WANDB_RUN=${MEG2SEM_SELECTION_WANDB_RUN:-pnpl/b2s2t/$MEG2SEM_RUN_ID}
MEG2SEM_CHECKPOINT_SELECTION=${MEG2SEM_CHECKPOINT_SELECTION:-priority}
MEG2SEM_CHECKPOINT=${MEG2SEM_CHECKPOINT:-}
MEG2SEM_OUTPUT_NORMALIZATION=${MEG2SEM_OUTPUT_NORMALIZATION:-auto}
MEG2SEM_PREDICTION_MODE=${MEG2SEM_PREDICTION_MODE:-full_ada}
MEG2SEM_RESIDUAL_COMPONENT_NPZ=${MEG2SEM_RESIDUAL_COMPONENT_NPZ:-}
MEG2SEM_RESIDUAL_MEAN_KEY=${MEG2SEM_RESIDUAL_MEAN_KEY:-train_mean}
MEG2SEM_RESIDUAL_TARGET_SCALE=${MEG2SEM_RESIDUAL_TARGET_SCALE:-1.0}
INIT_E2E_CHECKPOINT=${INIT_E2E_CHECKPOINT:-}

EVAL_ONLY=${EVAL_ONLY:-1}
TRAIN_MEG2SEM=${TRAIN_MEG2SEM:-0}
TRAIN_SEMANTIC_PROJECTOR=${TRAIN_SEMANTIC_PROJECTOR:-0}
UNFREEZE_ELF=${UNFREEZE_ELF:-0}
UNFREEZE_LAST_N_BLOCKS=${UNFREEZE_LAST_N_BLOCKS:-4}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-50}
LR=${LR:-1e-5}
MEG2SEM_LR=${MEG2SEM_LR:-$LR}
SEMANTIC_PROJECTOR_LR=${SEMANTIC_PROJECTOR_LR:-$LR}
ELF_LR=${ELF_LR:-1e-5}
TEACHER_CONTEXT_LOSS_WEIGHT=${TEACHER_CONTEXT_LOSS_WEIGHT:-0.0}
EVAL_EVERY=${EVAL_EVERY:-500}
RETRIEVAL_EVAL_EVERY=${RETRIEVAL_EVAL_EVERY:-500}
INTERFACE_MONITOR_EVERY=${INTERFACE_MONITOR_EVERY:-10}
INTERFACE_MONITOR_BATCH_SIZE=${INTERFACE_MONITOR_BATCH_SIZE:-16}
SAVE_EVAL_CHECKPOINTS=${SAVE_EVAL_CHECKPOINTS:-0}
EVAL_CHECKPOINT_TOP_K=${EVAL_CHECKPOINT_TOP_K:-3}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-32}
CFG_SCALE=${CFG_SCALE:-1.0}
SELF_COND_CFG_SCALE=${SELF_COND_CFG_SCALE:-1.0}
SEED=${SEED:-7}
VAL_NUM_EXAMPLES=${VAL_NUM_EXAMPLES:-512}
EVAL_NUM_EXAMPLES=${EVAL_NUM_EXAMPLES:-64}
RETRIEVAL_NUM_EXAMPLES=${RETRIEVAL_NUM_EXAMPLES:-64}
EVAL_START_INDEX=${EVAL_START_INDEX:-0}
RETRIEVAL_START_INDEX=${RETRIEVAL_START_INDEX:-$EVAL_START_INDEX}
RETRIEVAL_BATCH_SIZE=${RETRIEVAL_BATCH_SIZE:-128}

export SKIP_ARC_SETUP=${SKIP_ARC_SETUP:-1}
export WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
export WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-meg2sem_minilm_direct_elf}
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

if [[ -z "$MEG2SEM_CHECKPOINT" ]]; then
  if [[ ! -d "$MEG2SEM_CHECKPOINT_DIR" ]]; then
    echo "MEG2SEM_CHECKPOINT_DIR does not exist: $MEG2SEM_CHECKPOINT_DIR" >&2
    exit 2
  fi
  case "$MEG2SEM_CHECKPOINT_SELECTION" in
    priority|ndcg_dominant|balanced|loss)
      echo "Selecting MEG2SEM checkpoint with strategy=$MEG2SEM_CHECKPOINT_SELECTION from $MEG2SEM_SELECTION_WANDB_RUN"
      if selected_checkpoint=$(WANDB_SILENT=true python scripts/select_meg2sem_handoff_checkpoint.py \
        --wandb-run "$MEG2SEM_SELECTION_WANDB_RUN" \
        --checkpoint-dir "$MEG2SEM_CHECKPOINT_DIR" \
        --strategy "$MEG2SEM_CHECKPOINT_SELECTION" \
        --output path | tail -n 1); then
        MEG2SEM_CHECKPOINT="$selected_checkpoint"
      else
        echo "W&B checkpoint selection failed; falling back to MEG2SEM_RECOMMENDED_CHECKPOINT=$MEG2SEM_RECOMMENDED_CHECKPOINT" >&2
        MEG2SEM_CHECKPOINT="$MEG2SEM_RECOMMENDED_CHECKPOINT"
      fi
      ;;
    pinned)
      MEG2SEM_CHECKPOINT="$MEG2SEM_RECOMMENDED_CHECKPOINT"
      ;;
    latest)
      MEG2SEM_CHECKPOINT=$(find "$MEG2SEM_CHECKPOINT_DIR" -maxdepth 1 -type f -name '*.ckpt' -print | sort -V | tail -n 1)
      ;;
    *)
      echo "Unknown MEG2SEM_CHECKPOINT_SELECTION=$MEG2SEM_CHECKPOINT_SELECTION" >&2
      exit 2
      ;;
  esac
fi
if [[ -z "$MEG2SEM_CHECKPOINT" || ! -f "$MEG2SEM_CHECKPOINT" ]]; then
  echo "Could not resolve MEG2SEM_CHECKPOINT. Set MEG2SEM_CHECKPOINT explicitly." >&2
  exit 2
fi

echo "Running MEG2SEM MiniLM -> semantic projector -> ELF"
echo "TRAIN_NPZ=$TRAIN_NPZ"
echo "MEG2SEM_CHECKPOINT=$MEG2SEM_CHECKPOINT"
echo "MEG2SEM_CHECKPOINT_SELECTION=$MEG2SEM_CHECKPOINT_SELECTION"
echo "MEG2SEM_OUTPUT_NORMALIZATION=$MEG2SEM_OUTPUT_NORMALIZATION"
echo "MEG2SEM_PREDICTION_MODE=$MEG2SEM_PREDICTION_MODE"
echo "MEG2SEM_RESIDUAL_COMPONENT_NPZ=${MEG2SEM_RESIDUAL_COMPONENT_NPZ:-<none>}"
echo "MEG2SEM_RESIDUAL_MEAN_KEY=$MEG2SEM_RESIDUAL_MEAN_KEY"
echo "MEG2SEM_RESIDUAL_TARGET_SCALE=$MEG2SEM_RESIDUAL_TARGET_SCALE"
echo "ELF_CHECKPOINT=$ELF_CHECKPOINT"
echo "SEMANTIC_PROJECTOR_CHECKPOINT=$SEMANTIC_PROJECTOR_CHECKPOINT"
echo "INIT_E2E_CHECKPOINT=$INIT_E2E_CHECKPOINT"
echo "NUM_SAMPLING_STEPS=$NUM_SAMPLING_STEPS"
echo "CFG_SCALE=$CFG_SCALE SELF_COND_CFG_SCALE=$SELF_COND_CFG_SCALE SEED=$SEED"
echo "OUTPUT_DIR=$OUTPUT_DIR"

args=(
  scripts/train_npz_meg_to_frozen_elf.py
  --train-npz "$TRAIN_NPZ"
  --output_dir "$OUTPUT_DIR"
  --checkpoint_path "$SEMANTIC_PROJECTOR_CHECKPOINT"
  --elf-checkpoint-path "$ELF_CHECKPOINT"
  --semantic-projector-checkpoint "$SEMANTIC_PROJECTOR_CHECKPOINT"
  --adapter-kind meg2sem_projector
  --meg2sem-checkpoint "$MEG2SEM_CHECKPOINT"
  --meg2sem-output-normalization "$MEG2SEM_OUTPUT_NORMALIZATION"
  --meg2sem-prediction-mode "$MEG2SEM_PREDICTION_MODE"
  --meg-standardization none
  --meg-clip-boundary -1
  --device cuda
  --num-examples 0
  --val-num-examples "$VAL_NUM_EXAMPLES"
  --eval-num-examples "$EVAL_NUM_EXAMPLES"
  --retrieval-num-examples "$RETRIEVAL_NUM_EXAMPLES"
  --eval-start-index "$EVAL_START_INDEX"
  --retrieval-start-index "$RETRIEVAL_START_INDEX"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --eval_every "$EVAL_EVERY"
  --retrieval-eval-every "$RETRIEVAL_EVAL_EVERY"
  --retrieval-batch-size "$RETRIEVAL_BATCH_SIZE"
  --num_sampling_steps "$NUM_SAMPLING_STEPS"
  --context_length 64
  --semantic-hidden-dim 4096
  --teacher-context-loss-weight "$TEACHER_CONTEXT_LOSS_WEIGHT"
  --lr "$LR"
  --meg2sem-lr "$MEG2SEM_LR"
  --semantic-projector-lr "$SEMANTIC_PROJECTOR_LR"
  --weight_decay 1e-2
  --cfg_scale "$CFG_SCALE"
  --self_cond_cfg_scale "$SELF_COND_CFG_SCALE"
  --train-target-mask-mode full
  --interface-monitor-every "$INTERFACE_MONITOR_EVERY"
  --interface-monitor-batch-size "$INTERFACE_MONITOR_BATCH_SIZE"
  --generation-t5-retrieval
  --seed "$SEED"
  --use_wandb
  --wandb_project "$WANDB_PROJECT"
  --wandb_entity "$WANDB_ENTITY"
  --wandb_group "$WANDB_RUN_GROUP"
  --wandb_run_name "$WANDB_RUN_NAME"
  --wandb_sample_examples 16
  --wandb_notes "Direct MEG2SEM run $MEG2SEM_RUN_ID: predicted MiniLM vectors are passed through the frozen MiniLM->ELF K64 projector and evaluated/generated with the paired ELF checkpoint. Set EVAL_ONLY=0 TRAIN_MEG2SEM=1 and optionally UNFREEZE_ELF=1 for end-to-end fine-tuning."
)

if [[ "$MEG2SEM_PREDICTION_MODE" == "mean_residual" ]]; then
  if [[ -z "$MEG2SEM_RESIDUAL_COMPONENT_NPZ" || ! -f "$MEG2SEM_RESIDUAL_COMPONENT_NPZ" ]]; then
    echo "MEG2SEM_RESIDUAL_COMPONENT_NPZ is required for mean_residual prediction" >&2
    exit 2
  fi
  args+=(
    --meg2sem-residual-component-npz "$MEG2SEM_RESIDUAL_COMPONENT_NPZ"
    --meg2sem-residual-mean-key "$MEG2SEM_RESIDUAL_MEAN_KEY"
    --meg2sem-residual-target-scale "$MEG2SEM_RESIDUAL_TARGET_SCALE"
  )
fi

if [[ -n "$VAL_NPZ" ]]; then
  args+=(--val-npz "$VAL_NPZ")
fi

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
if [[ -n "$INIT_E2E_CHECKPOINT" ]]; then
  args+=(--init-e2e-checkpoint "$INIT_E2E_CHECKPOINT")
fi

python "${args[@]}"
