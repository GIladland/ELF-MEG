#!/usr/bin/env bash
#SBATCH --job-name=meg_wer_confirm
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-1
#SBATCH --output=submit-jobs/meg_minilm_elf_decode_confirm-%A_%a.out
#SBATCH --error=submit-jobs/meg_minilm_elf_decode_confirm-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
source_run=sherlock_sva_meg2sem_minilm_e2e_finetune_dt8z7o55
source_dir=/data/engs-pnpl/glandau/elf-runs/$source_run
oracle_run=sherlock_sva_meg2sem_corpus_minilm_oracle_overfit_full_k64_ep50_b64_20260828
oracle_checkpoint=/data/engs-pnpl/glandau/elf-runs/$oracle_run/best.pt

case "$SLURM_ARRAY_TASK_ID" in
  0) sampling_steps=64; cfg_scale=0.5; objective=wer ;;
  1) sampling_steps=32; cfg_scale=1.0; objective=top1 ;;
  *) echo "Unsupported task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

checkpoint=$(find "$source_dir/checkpoints" -maxdepth 1 -type f -name 'eval_step_00000600_score_*.pt' -print -quit)
if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
  echo "Missing E2E checkpoint for step 600 in $source_dir/checkpoints" >&2
  exit 2
fi

export EVAL_START_INDEX=${EVAL_START_INDEX:-64}
export RETRIEVAL_START_INDEX=${RETRIEVAL_START_INDEX:-$EVAL_START_INDEX}
export EVAL_NUM_EXAMPLES=${EVAL_NUM_EXAMPLES:-256}
export RETRIEVAL_NUM_EXAMPLES=${RETRIEVAL_NUM_EXAMPLES:-$EVAL_NUM_EXAMPLES}
export SEED=${SEED:-7}
export RUN_TAG=${RUN_TAG:-meg_minilm_elf_decode_confirm_${objective}_${source_run}_step600_s${sampling_steps}_cfg${cfg_scale}_seed${SEED}_offset${EVAL_START_INDEX}_n${EVAL_NUM_EXAMPLES}_20260829}
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export ELF_CHECKPOINT=$oracle_checkpoint
export SEMANTIC_PROJECTOR_CHECKPOINT=$oracle_checkpoint
export INIT_E2E_CHECKPOINT=$checkpoint
export MEG2SEM_CHECKPOINT=/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/dt8z7o55/checkpoints/sherlock_sva_minilm_books1to9_to_sherlock12_bs60_seed49-epoch=999-val_loss=27.19.ckpt
export MEG2SEM_OUTPUT_NORMALIZATION=auto
export EVAL_ONLY=1
export TRAIN_MEG2SEM=0
export TRAIN_SEMANTIC_PROJECTOR=0
export UNFREEZE_ELF=0
export VAL_NUM_EXAMPLES=512
export RETRIEVAL_BATCH_SIZE=128
export NUM_SAMPLING_STEPS=$sampling_steps
export CFG_SCALE=$cfg_scale
export SELF_COND_CFG_SCALE=1.0
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export WANDB_RUN_GROUP=meg_minilm_elf_decode_disjoint_confirm_20260829
export WANDB_RUN_NAME=$RUN_TAG

exec bash "$PROJECT_ROOT/submit-jobs/sherlock_sva_meg2sem_minilm_direct_elf_dt8z7o55.sbatch"
