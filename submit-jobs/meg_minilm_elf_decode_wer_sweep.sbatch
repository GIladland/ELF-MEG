#!/usr/bin/env bash
#SBATCH --job-name=meg_wer_sweep
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-11
#SBATCH --output=submit-jobs/meg_minilm_elf_decode_wer_sweep-%A_%a.out
#SBATCH --error=submit-jobs/meg_minilm_elf_decode_wer_sweep-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
source_run=sherlock_sva_meg2sem_minilm_e2e_finetune_dt8z7o55
source_dir=/data/engs-pnpl/glandau/elf-runs/$source_run
oracle_run=sherlock_sva_meg2sem_corpus_minilm_oracle_overfit_full_k64_ep50_b64_20260828
oracle_checkpoint=/data/engs-pnpl/glandau/elf-runs/$oracle_run/best.pt

checkpoint_slot=$((SLURM_ARRAY_TASK_ID / 6))
config_slot=$((SLURM_ARRAY_TASK_ID % 6))
case "$checkpoint_slot" in
  0) checkpoint_step=600 ;;
  1) checkpoint_step=1250 ;;
  *) echo "Unsupported checkpoint slot: $checkpoint_slot" >&2; exit 2 ;;
esac
case "$config_slot" in
  0) sampling_steps=16; cfg_scale=0.5 ;;
  1) sampling_steps=16; cfg_scale=1.0 ;;
  2) sampling_steps=32; cfg_scale=0.5 ;;
  3) sampling_steps=32; cfg_scale=1.0 ;;
  4) sampling_steps=64; cfg_scale=0.5 ;;
  5) sampling_steps=64; cfg_scale=1.0 ;;
  *) echo "Unsupported config slot: $config_slot" >&2; exit 2 ;;
esac

checkpoint=$(find "$source_dir/checkpoints" -maxdepth 1 -type f -name "eval_step_*$(printf '%05d' "$checkpoint_step")_score_*.pt" -print -quit)
if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
  echo "Missing E2E checkpoint for step $checkpoint_step in $source_dir/checkpoints" >&2
  exit 2
fi

export RUN_TAG=meg_minilm_elf_decode_wer_${source_run}_step${checkpoint_step}_s${sampling_steps}_cfg${cfg_scale}_seed7_20260829
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
export EVAL_NUM_EXAMPLES=64
export RETRIEVAL_NUM_EXAMPLES=64
export RETRIEVAL_BATCH_SIZE=128
export NUM_SAMPLING_STEPS=$sampling_steps
export CFG_SCALE=$cfg_scale
export SELF_COND_CFG_SCALE=1.0
export SEED=7
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export WANDB_RUN_GROUP=meg_minilm_elf_decode_wer_sweep_20260829
export WANDB_RUN_NAME=$RUN_TAG

exec bash "$PROJECT_ROOT/submit-jobs/sherlock_sva_meg2sem_minilm_direct_elf_dt8z7o55.sbatch"
