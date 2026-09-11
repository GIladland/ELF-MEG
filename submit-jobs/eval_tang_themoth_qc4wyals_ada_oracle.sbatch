#!/usr/bin/env bash
#SBATCH --job-name=qc4_ada_elf
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --array=0-1%2
#SBATCH --output=submit-jobs/eval_tang_themoth_qc4wyals_ada_oracle-%A_%a.out
#SBATCH --error=submit-jobs/eval_tang_themoth_qc4wyals_ada_oracle-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
data_dir=$DATA_ROOT/tang_themoth_qc4wyals_ada_v1
oracle_run=/data/engs-pnpl/glandau/elf-runs/fmri_knowntext_ada002_elfb_long_ep300_testfirst_k64_seed49_20260830
init_checkpoint=${INIT_E2E_CHECKPOINT_OVERRIDE:-$oracle_run/best.pt}
run_suffix=${RUN_SUFFIX_OVERRIDE:-existing_elfb_20260901}

case "$SLURM_ARRAY_TASK_ID" in
  0)
    label=exact_oracle
    source_npz=$data_dir/exact_ada002_birthofanation26.npz
    ;;
  1)
    label=qc4wyals_predicted
    source_npz=$data_dir/qc4wyals_predicted_ada002_birthofanation26.npz
    ;;
  *) exit 2 ;;
esac

export RUN_TAG=tang_themoth_birthofanation26_ada002_${label}_${run_suffix}
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export SOURCE_NPZ=$source_npz
export NPZ_PATH=$source_npz
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export MODEL=ELF-B
export CHECKPOINT_PATH=embedded-language-flows/ELF-B-owt-torch
export INIT_E2E_CHECKPOINT=$init_checkpoint
export INIT_E2E_ADAPTER_MISMATCH=error
export SEMANTIC_ADAPTER_KIND=flat
export SEMANTIC_INPUT_PROJECTION_DIM=0
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export ADAPTER_DROPOUT=0.0
export FREEZE_ELF=1
export FREEZE_SEMANTIC_ADAPTER=1
if [[ -n "${ELF_LORA_TARGETS_OVERRIDE:-}" ]]; then
  export ELF_LORA_TARGETS=${ELF_LORA_TARGETS_OVERRIDE//:/,}
fi
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=26
export RETRIEVAL_NUM_EXAMPLES=26
export GENERATION_T5_RETRIEVAL=1
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export EVAL_ONLY=1
export EPOCHS=0
export BATCH_SIZE=16
export SEED=49
export USE_WANDB=0
export SAVE_FINAL_CHECKPOINT=0
export SAVE_EVAL_CHECKPOINTS=0
export TARGET_LATENTS_CACHE=$data_dir/${label}_t5small_latents_fp16.npy
export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1
export WANDB_NOTES="Zero-shot ADA-to-ELF audit for qc4wyals. Task 0 uses exact normalized ADA; task 1 uses qc4wyals normalized brain predictions. The frozen ELF checkpoint was trained without either qc4wyals predictions or Birth of a Nation brain rows."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
