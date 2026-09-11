#!/usr/bin/env bash
#SBATCH --job-name=sva_mlm_oracle
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=submit-jobs/sherlock_sva_meg2sem_corpus_minilm_oracle_overfit-%j.out
#SBATCH --error=submit-jobs/sherlock_sva_meg2sem_corpus_minilm_oracle_overfit-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}

export PROJECT_ROOT DATA_ROOT
export SOURCE_NPZ=${SOURCE_NPZ:-$DATA_ROOT/sherlock_sva/meg2sem/sherlock1to9_sva_no_coord_decl5to18_train_minilm_meg2sem_segment3000_fp16.npz}
export NPZ_PATH=${NPZ_PATH:-$SOURCE_NPZ}
export INPUT_KEY=${INPUT_KEY:-input_embeddings}
export SENTENCE_KEY=${SENTENCE_KEY:-sentence}

export RUN_TAG=${RUN_TAG:-sherlock_sva_meg2sem_corpus_minilm_oracle_overfit_full_k64_ep50_b64_20260828}
export OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/$RUN_TAG}
export TARGET_LATENTS_CACHE=${TARGET_LATENTS_CACHE:-$DATA_ROOT/sherlock_sva/meg2sem/${RUN_TAG}_t5small_latents_fp16.npy}
export INIT_E2E_CHECKPOINT=${INIT_E2E_CHECKPOINT:-/data/engs-pnpl/glandau/elf-runs/sherlock1to9_sva_no_coord_decl5to18_plus_oraclevalaug64x_to_sherlock12_sva_no_coord_decl5to18_minilm_k64_epoch300_fullmask_long2d/best.pt}
export INIT_E2E_ADAPTER_MISMATCH=${INIT_E2E_ADAPTER_MISMATCH:-error}

# All 10,930 rows are deliberately used for training. Evaluation samples come
# from the same rows and measure the oracle diffusion ceiling, not generalization.
export NUM_EXAMPLES=${NUM_EXAMPLES:-0}
export VAL_NUM_EXAMPLES=${VAL_NUM_EXAMPLES:-0}
export EPOCHS=${EPOCHS:-50}
export BATCH_SIZE=${BATCH_SIZE:-64}
export EVAL_EVERY=${EVAL_EVERY:-1000}
export EVAL_NUM_EXAMPLES=${EVAL_NUM_EXAMPLES:-256}
export RETRIEVAL_EVAL_EVERY=${RETRIEVAL_EVAL_EVERY:-1000}
export RETRIEVAL_NUM_EXAMPLES=${RETRIEVAL_NUM_EXAMPLES:-512}
export RETRIEVAL_BATCH_SIZE=${RETRIEVAL_BATCH_SIZE:-256}
export NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-32}
export TRAIN_TIMESTEP_STEPS=${TRAIN_TIMESTEP_STEPS:-32}

export CONTEXT_LENGTH=${CONTEXT_LENGTH:-64}
export SEMANTIC_HIDDEN_DIM=${SEMANTIC_HIDDEN_DIM:-4096}
export SEMANTIC_ADAPTER_KIND=${SEMANTIC_ADAPTER_KIND:-flat}
export SEMANTIC_INPUT_PROJECTION_DIM=${SEMANTIC_INPUT_PROJECTION_DIM:-0}
export ADAPTER_DROPOUT=${ADAPTER_DROPOUT:-0.0}
export FREEZE_ELF=${FREEZE_ELF:-0}
export LAST_N_BLOCKS=${LAST_N_BLOCKS:--1}
export LR=${LR:-2e-5}
export TRAIN_TARGET_MASK_MODE=${TRAIN_TARGET_MASK_MODE:-full}
export COND_DROPOUT_PROB=${COND_DROPOUT_PROB:-0.0}
export SAVE_FINAL_CHECKPOINT=${SAVE_FINAL_CHECKPOINT:-1}

export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-sherlock_sva_minilm_oracle_overfit}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$RUN_TAG}
export WANDB_NOTES=${WANDB_NOTES:-Exact-corpus oracle overfit ceiling for the 10,930-row Sherlock SVA MEG2SEM MiniLM archive. Warm-start the proven MiniLM-to-ELF bridge, then fine-tune the full ELF and semantic adapter on all rows before evaluating MEG2SEM-predicted MiniLM vectors.}

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
