#!/usr/bin/env bash
#SBATCH --job-name=fmri_pres_test
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-2
#SBATCH --output=submit-jobs/eval_fmri_freshmem_semantic_preservation_test107-%A_%a.out
#SBATCH --error=submit-jobs/eval_fmri_freshmem_semantic_preservation_test107-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
snapshot=${ORACLE_SNAPSHOT:?Set ORACLE_SNAPSHOT}
snapshot_tag=${ORACLE_SNAPSHOT_TAG:-step024000}
winner_checkpoint=${WINNER_CHECKPOINT:?Set WINNER_CHECKPOINT}
winner_variant=${WINNER_VARIANT:?Set WINNER_VARIANT}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
source_dir=$DATA_ROOT/minilm/oracle1536_delayed_seg10_middle10content_clockfix

export NPZ_PATH=$source_dir/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_pred1536_test_middle10content_clockfix.npz
export SOURCE_NPZ=$NPZ_PATH
export SEMANTIC_ALIGNMENT_TARGET_NPZ=$known_dir/oracle_exact_minilm384_test107.npz
export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings
export SEMANTIC_ADAPTER_KIND=residual_mlp
export SEMANTIC_MAPPER_HIDDEN_DIM=512
export SEMANTIC_DELAY_TOKENS=4
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export ADAPTER_DROPOUT=0.0
export FREEZE_ELF=1
export ELF_LORA_RANK=0
export ELF_LORA_ALPHA=8
export ELF_LORA_DROPOUT=0.0
export ELF_LORA_TARGETS=attention,mlp
export ELF_LORA_LAST_N_BLOCKS=2

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    eval_name=raw_residual
    export INIT_E2E_CHECKPOINT=$snapshot
    export INIT_E2E_ADAPTER_MISMATCH=residual-mlp
    ;;
  1)
    eval_name=$winner_variant
    export INIT_E2E_CHECKPOINT=$winner_checkpoint
    export INIT_E2E_ADAPTER_MISMATCH=error
    if [[ "$winner_variant" == *lora4last2* ]]; then
      export ELF_LORA_RANK=4
    fi
    ;;
  2)
    eval_name=oracle_exact_minilm384
    export NPZ_PATH=$known_dir/oracle_exact_minilm384_test107.npz
    export SOURCE_NPZ=$NPZ_PATH
    export SEMANTIC_ADAPTER_KIND=flat
    export INIT_E2E_CHECKPOINT=$snapshot
    export INIT_E2E_ADAPTER_MISMATCH=error
    ;;
  *) exit 2 ;;
esac

export RUN_TAG=eval_fmri_freshmem_semantic_preservation_${snapshot_tag}_${eval_name}_test107_seed49_20260901
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$known_dir/eval_latents/${RUN_TAG}_t5small_latents_fp16.npy
export EVAL_ONLY=1
export EPOCHS=0
export NUM_EXAMPLES=107
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=107
export RETRIEVAL_NUM_EXAMPLES=107
export BATCH_SIZE=32
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CFG_SCALE=1.0
export SELF_COND_CFG_SCALE=1.0
export GENERATION_T5_RETRIEVAL=1
export CHECKPOINT_METRIC=generation_t5_top1
export SEED=49

export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=fmri_freshmem_semantic_preservation_20260901
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Frozen one-time test107 evaluation for $eval_name after retrieval-gated val266 selection. Brain-predicted test vectors were absent from all training and selection; the oracle arm measures the deliberately text-memorized MiniLM-to-ELF ceiling."

export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
