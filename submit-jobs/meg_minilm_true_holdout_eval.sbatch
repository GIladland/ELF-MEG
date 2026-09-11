#!/usr/bin/env bash
#SBATCH --job-name=meg_mini_true_ho
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-1
#SBATCH --output=submit-jobs/meg_minilm_true_holdout_eval-%A_%a.out
#SBATCH --error=submit-jobs/meg_minilm_true_holdout_eval-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
source_dir=$DATA_ROOT/sherlock_sva/meg2sem
holdout_split=${HOLDOUT_SPLIT:-val}
decode_variant=${DECODE_VARIANT:-$SLURM_ARRAY_TASK_ID}

case "$decode_variant" in
  0) sampling_steps=64; cfg_scale=0.5; objective=wer ;;
  1) sampling_steps=32; cfg_scale=1.0; objective=top1 ;;
  *) echo "Unsupported DECODE_VARIANT=$decode_variant" >&2; exit 2 ;;
esac

case "$holdout_split" in
  val)
    export VAL_NPZ=$source_dir/sherlock1_ses11_run2_sva_no_coord_decl5to18_val_minilm_meg2sem_segment3000_fp16.npz
    eval_n=48
    ;;
  test)
    export VAL_NPZ=$source_dir/sherlock1_ses12_run2_sva_no_coord_decl5to18_test_minilm_meg2sem_segment3000_fp16.npz
    eval_n=27
    ;;
  *) echo "Unsupported HOLDOUT_SPLIT=$holdout_split" >&2; exit 2 ;;
esac

export TRAIN_NPZ=$source_dir/sherlock1to9_sva_no_coord_decl5to18_train_minilm_meg2sem_segment3000_fp16.npz
oracle_run=sherlock_sva_meg2sem_corpus_minilm_oracle_overfit_full_k64_ep50_b64_20260828
oracle_checkpoint=/data/engs-pnpl/glandau/elf-runs/$oracle_run/best.pt
export ELF_CHECKPOINT=$oracle_checkpoint
export SEMANTIC_PROJECTOR_CHECKPOINT=$oracle_checkpoint
export INIT_E2E_CHECKPOINT=
export MEG2SEM_CHECKPOINT=${MEG2SEM_CHECKPOINT:-/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/dt8z7o55/checkpoints/sherlock_sva_minilm_books1to9_to_sherlock12_bs60_seed49-epoch=49-val_loss=18.51.ckpt}
export MEG2SEM_OUTPUT_NORMALIZATION=auto
export EVAL_ONLY=1
export TRAIN_MEG2SEM=0
export TRAIN_SEMANTIC_PROJECTOR=0
export UNFREEZE_ELF=0
export EVAL_START_INDEX=0
export RETRIEVAL_START_INDEX=0
export EVAL_NUM_EXAMPLES=$eval_n
export RETRIEVAL_NUM_EXAMPLES=$eval_n
export RETRIEVAL_BATCH_SIZE=128
export INTERFACE_MONITOR_BATCH_SIZE=$eval_n
export NUM_SAMPLING_STEPS=$sampling_steps
export CFG_SCALE=$cfg_scale
export SELF_COND_CFG_SCALE=1.0
export SEED=7
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export RUN_TAG=${RUN_TAG:-meg_minilm_TRUE_BRAIN_HOLDOUT_${holdout_split}_${objective}_epoch49_s${sampling_steps}_cfg${cfg_scale}_seed7_n${eval_n}_20260829}
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export WANDB_RUN_GROUP=meg_minilm_true_brain_holdout_20260829
export WANDB_RUN_NAME=$RUN_TAG

exec bash "$PROJECT_ROOT/submit-jobs/sherlock_sva_meg2sem_minilm_direct_elf_dt8z7o55.sbatch"
