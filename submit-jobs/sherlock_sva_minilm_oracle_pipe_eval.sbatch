#!/usr/bin/env bash
#SBATCH --job-name=sh_mlm_pipe
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=submit-jobs/sherlock_sva_minilm_oracle_pipe_eval-%j.out
#SBATCH --error=submit-jobs/sherlock_sva_minilm_oracle_pipe_eval-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}

RUN_TAG=${RUN_TAG:-sherlock_sva_minilm_oracle_pipe_base_minilm_k64}
NPZ_PATH=${NPZ_PATH:-$DATA_ROOT/sherlock_sva/meg2sem/sherlock1to9_sva_no_coord_decl5to18_train_minilm_meg2sem_segment3000_fp16.npz}
INPUT_KEY=${INPUT_KEY:-input_embeddings}
ELF_CHECKPOINT=${ELF_CHECKPOINT:-/data/engs-pnpl/glandau/elf-runs/sherlock1to9_sva_no_coord_decl5to18_plus_oraclevalaug64x_to_sherlock12_sva_no_coord_decl5to18_minilm_k64_epoch300_fullmask_long2d/best.pt}
INIT_E2E_CHECKPOINT=${INIT_E2E_CHECKPOINT:-$ELF_CHECKPOINT}
OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/$RUN_TAG}

VAL_NUM_EXAMPLES=${VAL_NUM_EXAMPLES:-512}
EVAL_NUM_EXAMPLES=${EVAL_NUM_EXAMPLES:-64}
RETRIEVAL_NUM_EXAMPLES=${RETRIEVAL_NUM_EXAMPLES:-64}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-32}
RETRIEVAL_BATCH_SIZE=${RETRIEVAL_BATCH_SIZE:-128}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}

export PROJECT_ROOT
export DATA_ROOT
export NPZ_PATH
export OUTPUT_DIR
export SKIP_ARC_SETUP=${SKIP_ARC_SETUP:-1}
export WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
export WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-minilm_oracle_pipe}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$RUN_TAG}
export WANDB_DIR=${WANDB_DIR:-$DATA_ROOT/wandb}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-$DATA_ROOT/wandb-data}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$DATA_ROOT/wandb-cache}
export TMPDIR=${TMPDIR:-$DATA_ROOT/tmp}

echo "Running Sherlock MiniLM oracle pipe eval"
echo "NPZ_PATH=$NPZ_PATH"
echo "INPUT_KEY=$INPUT_KEY"
echo "ELF_CHECKPOINT=$ELF_CHECKPOINT"
echo "INIT_E2E_CHECKPOINT=$INIT_E2E_CHECKPOINT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "NUM_SAMPLING_STEPS=$NUM_SAMPLING_STEPS"

bash "$PROJECT_ROOT/scripts/arc_run_npz_semantic_to_elf.sh" \
  --input-key "$INPUT_KEY" \
  --checkpoint_path "$ELF_CHECKPOINT" \
  --init-e2e-checkpoint "$INIT_E2E_CHECKPOINT" \
  --eval-only \
  --freeze-elf \
  --device cuda \
  --num-examples 0 \
  --val-num-examples "$VAL_NUM_EXAMPLES" \
  --eval-num-examples "$EVAL_NUM_EXAMPLES" \
  --retrieval-num-examples "$RETRIEVAL_NUM_EXAMPLES" \
  --retrieval-batch-size "$RETRIEVAL_BATCH_SIZE" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --num_sampling_steps "$NUM_SAMPLING_STEPS" \
  --context_length 64 \
  --semantic-hidden-dim 4096 \
  --cfg_scale 1.0 \
  --self_cond_cfg_scale 1.0 \
  --train-target-mask-mode full \
  --target-latents-mode lazy \
  --generation-t5-retrieval \
  --seed 7 \
  --wandb_sample_examples 16 \
  --wandb_notes "Oracle diagnostic: feed true Sherlock MiniLM vectors directly into the MiniLM->ELF projector, bypassing MEG, to measure whether brain/e2e updates preserve or warp the normal semantic interface."
