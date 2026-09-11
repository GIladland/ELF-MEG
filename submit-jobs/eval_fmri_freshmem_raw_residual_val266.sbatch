#!/usr/bin/env bash
#SBATCH --job-name=fmri_pres_raw
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=submit-jobs/eval_fmri_freshmem_raw_residual_val266-%j.out
#SBATCH --error=submit-jobs/eval_fmri_freshmem_raw_residual_val266-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
snapshot=${ORACLE_SNAPSHOT:?Set ORACLE_SNAPSHOT}
snapshot_tag=${ORACLE_SNAPSHOT_TAG:-step024000}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1

export NPZ_PATH=$DATA_ROOT/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz
export SOURCE_NPZ=$NPZ_PATH
export SEMANTIC_ALIGNMENT_TARGET_NPZ=$known_dir/oofmix_exact_minilm384_targets_train23450_val266.npz
export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings
export TARGET_LATENTS_CACHE=$known_dir/oofmix_current10_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=$snapshot
export INIT_E2E_ADAPTER_MISMATCH=residual-mlp
export RUN_TAG=fmri_freshmem_semantic_preservation_${snapshot_tag}_raw_residual_val266_seed49_20260901
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG

export EVAL_ONLY=1
export EPOCHS=0
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=266
export EVAL_NUM_EXAMPLES=266
export RETRIEVAL_NUM_EXAMPLES=266
export BATCH_SIZE=32
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_ADAPTER_KIND=residual_mlp
export SEMANTIC_MAPPER_HIDDEN_DIM=512
export SEMANTIC_DELAY_TOKENS=4
export ADAPTER_DROPOUT=0.0
export FREEZE_ELF=1
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
export WANDB_NOTES="Frozen raw four-delay-mean validation baseline for the retrieval-preserving residual experiment. No optimization; val266 only; test107 absent."

export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
