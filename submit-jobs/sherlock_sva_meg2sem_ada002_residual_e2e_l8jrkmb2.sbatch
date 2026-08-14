#!/usr/bin/env bash
#SBATCH --job-name=sva_resid_e2e
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=submit-jobs/sherlock_sva_meg2sem_ada002_residual_e2e_l8jrkmb2-%j.out
#SBATCH --error=submit-jobs/sherlock_sva_meg2sem_ada002_residual_e2e_l8jrkmb2-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
PYTHON_BIN=${PYTHON_BIN:-/data/engs-pnpl/glandau/BrainDiffusion/elf-env/bin/python}

RUN_TAG=${RUN_TAG:-sherlock_sva_meg2sem_ada002_seg8000_l8jrkmb2_epoch21_residual_full_e2e}
TRAIN_NPZ=${TRAIN_NPZ:-$DATA_ROOT/sherlock_sva/meg2sem/sherlock1to9_sva_no_coord_decl5to18_train_ada002_meg2sem_segment8000_fp16.npz}
VAL_NPZ=${VAL_NPZ:-$DATA_ROOT/sherlock_sva/meg2sem/sherlock1_ses11plus12_sva_no_coord_decl5to18_valtest_ada002_meg2sem_segment8000_fp16.npz}
ADA_ELF_CHECKPOINT=${ADA_ELF_CHECKPOINT:-/data/engs-pnpl/glandau/elf-runs/sherlock1to9_sva_no_coord_decl5to18_plus_oraclevalaug64x_to_sherlock12_sva_no_coord_decl5to18_ada002_k64_epoch300_fullmask_long2d/best.pt}
ELF_CHECKPOINT=${ELF_CHECKPOINT:-$ADA_ELF_CHECKPOINT}
SEMANTIC_PROJECTOR_CHECKPOINT=${SEMANTIC_PROJECTOR_CHECKPOINT:-$ADA_ELF_CHECKPOINT}
MEG2SEM_CHECKPOINT=${MEG2SEM_CHECKPOINT:-/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/l8jrkmb2/checkpoints/sherlock_sva_ada002_seg8000_resid_scaled_resonly_rescos1_mse500_norm1_lr3e4_bs36_seed58_r1-epoch=21-val_recon_ada_nDCG_epoch=0.2976.ckpt}
RESIDUAL_COMPONENT_NPZ=${RESIDUAL_COMPONENT_NPZ:-$DATA_ROOT/sherlock_sva/ada_residual_components/mean_only_books1to9_sva_ada002_segment8000.npz}
RESIDUAL_TARGET_SCALE=${RESIDUAL_TARGET_SCALE:-2.073361729178968}
INIT_E2E_CHECKPOINT=${INIT_E2E_CHECKPOINT:-}
OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/$RUN_TAG}

EVAL_ONLY=${EVAL_ONLY:-0}
TRAIN_MEG2SEM=${TRAIN_MEG2SEM:-1}
TRAIN_SEMANTIC_PROJECTOR=${TRAIN_SEMANTIC_PROJECTOR:-1}
UNFREEZE_ELF=${UNFREEZE_ELF:-1}
UNFREEZE_LAST_N_BLOCKS=${UNFREEZE_LAST_N_BLOCKS:-4}
SAVE_EVAL_CHECKPOINTS=${SAVE_EVAL_CHECKPOINTS:-1}
EVAL_CHECKPOINT_TOP_K=${EVAL_CHECKPOINT_TOP_K:-3}
ORACLE_SEMANTIC_EVAL=${ORACLE_SEMANTIC_EVAL:-1}
SEMANTIC_ALIGNMENT_LOSS_WEIGHT=${SEMANTIC_ALIGNMENT_LOSS_WEIGHT:-0.25}
SEMANTIC_ALIGNMENT_LOSS_TYPE=${SEMANTIC_ALIGNMENT_LOSS_TYPE:-cosine}

BATCH_SIZE=${BATCH_SIZE:-12}
EPOCHS=${EPOCHS:-0}
STEPS=${STEPS:-2400}
NUM_EXAMPLES=${NUM_EXAMPLES:-0}
LR=${LR:-1e-5}
MEG2SEM_LR=${MEG2SEM_LR:-3e-6}
SEMANTIC_PROJECTOR_LR=${SEMANTIC_PROJECTOR_LR:-1e-5}
ELF_LR=${ELF_LR:-5e-6}
ELF_ATTN_DROPOUT=${ELF_ATTN_DROPOUT:-0.05}
ELF_PROJ_DROPOUT=${ELF_PROJ_DROPOUT:-0.05}
COND_DROPOUT_PROB=${COND_DROPOUT_PROB:-0}
DENOISER_LOSS_WEIGHT=${DENOISER_LOSS_WEIGHT:-1.0}
DECODER_LOSS_WEIGHT=${DECODER_LOSS_WEIGHT:-2.0}
EVAL_EVERY=${EVAL_EVERY:-300}
RETRIEVAL_EVAL_EVERY=${RETRIEVAL_EVAL_EVERY:-300}
INTERFACE_MONITOR_EVERY=${INTERFACE_MONITOR_EVERY:-50}
INTERFACE_MONITOR_BATCH_SIZE=${INTERFACE_MONITOR_BATCH_SIZE:-32}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-32}
EVAL_NUM_EXAMPLES=${EVAL_NUM_EXAMPLES:-75}
RETRIEVAL_NUM_EXAMPLES=${RETRIEVAL_NUM_EXAMPLES:-75}
TARGET_ENCODE_BATCH_SIZE=${TARGET_ENCODE_BATCH_SIZE:-64}
SKIP_EVAL_RETRIEVAL=${SKIP_EVAL_RETRIEVAL:-0}
GENERATION_T5_RETRIEVAL=${GENERATION_T5_RETRIEVAL:-1}
USE_WANDB=${USE_WANDB:-1}

export WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
export WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-meg2sem_ada002_residual_e2e}
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

for required in "$PYTHON_BIN" "$TRAIN_NPZ" "$VAL_NPZ" "$ELF_CHECKPOINT" "$SEMANTIC_PROJECTOR_CHECKPOINT" "$MEG2SEM_CHECKPOINT" "$RESIDUAL_COMPONENT_NPZ"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 2
  fi
done
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 127
fi

echo "Running scaled residual MEG2SEM -> reconstructed ADA -> ADA projector -> ELF"
echo "RUN_TAG=$RUN_TAG"
echo "TRAIN_NPZ=$TRAIN_NPZ"
echo "VAL_NPZ=$VAL_NPZ"
echo "MEG2SEM_CHECKPOINT=$MEG2SEM_CHECKPOINT"
echo "RESIDUAL_COMPONENT_NPZ=$RESIDUAL_COMPONENT_NPZ"
echo "RESIDUAL_TARGET_SCALE=$RESIDUAL_TARGET_SCALE"
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
  --meg2sem-output-normalization never
  --meg2sem-prediction-mode mean_residual
  --meg2sem-residual-component-npz "$RESIDUAL_COMPONENT_NPZ"
  --meg2sem-residual-mean-key train_mean
  --meg2sem-residual-target-scale "$RESIDUAL_TARGET_SCALE"
  --meg-standardization none
  --meg-clip-boundary -1
  --device cuda
  --num-examples "$NUM_EXAMPLES"
  --eval-num-examples "$EVAL_NUM_EXAMPLES"
  --retrieval-num-examples "$RETRIEVAL_NUM_EXAMPLES"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --steps "$STEPS"
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
  --semantic-alignment-loss-weight "$SEMANTIC_ALIGNMENT_LOSS_WEIGHT"
  --semantic-alignment-loss-type "$SEMANTIC_ALIGNMENT_LOSS_TYPE"
  --lr "$LR"
  --meg2sem-lr "$MEG2SEM_LR"
  --semantic-projector-lr "$SEMANTIC_PROJECTOR_LR"
  --weight_decay 1e-2
  --cfg_scale 1.0
  --self_cond_cfg_scale 1.0
  --cond_dropout_prob "$COND_DROPOUT_PROB"
  --denoiser_loss_weight "$DENOISER_LOSS_WEIGHT"
  --decoder_loss_weight "$DECODER_LOSS_WEIGHT"
  --elf-attn-dropout "$ELF_ATTN_DROPOUT"
  --elf-proj-dropout "$ELF_PROJ_DROPOUT"
  --train-target-mask-mode full
  --interface-monitor-every "$INTERFACE_MONITOR_EVERY"
  --interface-monitor-batch-size "$INTERFACE_MONITOR_BATCH_SIZE"
  --seed 7
  --wandb_sample_examples 16
  --wandb_notes "Residual E2E bridge. MEG2SEM l8jrkmb2 epoch21 outputs scaled mean-only ADA residuals; BrainDiffusion reconstructs ADA as normalize(train_mean + output / 2.073361729178968), then routes through the ADA oracle projector and fine-tunes MEG2SEM, projector, and selected ELF blocks."
)

if [[ -n "$INIT_E2E_CHECKPOINT" ]]; then
  args+=(--init-e2e-checkpoint "$INIT_E2E_CHECKPOINT")
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
if [[ "$ORACLE_SEMANTIC_EVAL" == "1" ]]; then
  args+=(--oracle-semantic-eval)
fi
if [[ "$SKIP_EVAL_RETRIEVAL" == "1" ]]; then
  args+=(--skip-eval-retrieval)
fi
if [[ "$GENERATION_T5_RETRIEVAL" == "1" ]]; then
  args+=(--generation-t5-retrieval)
fi
if [[ "$USE_WANDB" == "1" ]]; then
  args+=(--use_wandb --wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY" --wandb_group "$WANDB_RUN_GROUP" --wandb_run_name "$WANDB_RUN_NAME")
fi

"$PYTHON_BIN" "${args[@]}"
