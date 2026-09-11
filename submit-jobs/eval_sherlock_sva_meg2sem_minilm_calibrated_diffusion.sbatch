#!/usr/bin/env bash
#SBATCH --job-name=sva_mlm_mapev
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_cc:7.0|gpu_cc:8.0|gpu_cc:8.9|gpu_cc:9.0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/eval_sherlock_sva_meg2sem_minilm_calibrated_diffusion-%j.out
#SBATCH --error=submit-jobs/eval_sherlock_sva_meg2sem_minilm_calibrated_diffusion-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
RUN_TAG=${RUN_TAG:-sherlock_sva_meg2sem_dt8z7o55_minilm_traincal_val512_test512}
NPZ_PATH=${NPZ_PATH:-$DATA_ROOT/sherlock_sva/meg2sem/calibrated/$RUN_TAG/mapped_minilm384.npz}
ELF_RUN_TAG=${ELF_RUN_TAG:-sherlock_sva_meg2sem_corpus_minilm_oracle_overfit_full_k64_ep50_b64_20260828}
ELF_CHECKPOINT=${ELF_CHECKPOINT:-/data/engs-pnpl/glandau/elf-runs/$ELF_RUN_TAG/best.pt}

if [[ ! -f "$NPZ_PATH" ]]; then
  echo "Mapped MiniLM NPZ is missing: $NPZ_PATH" >&2
  exit 2
fi

export PROJECT_ROOT DATA_ROOT NPZ_PATH ELF_CHECKPOINT
export INIT_E2E_CHECKPOINT=$ELF_CHECKPOINT
export OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/${RUN_TAG}_diffusion_eval512}
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export PYTHON_BIN=${PYTHON_BIN:-$TMP_ENV_ROOT/bin/python}
export SKIP_ARC_SETUP=1
export VAL_NUM_EXAMPLES=512
export EVAL_NUM_EXAMPLES=${EVAL_NUM_EXAMPLES:-256}
export RETRIEVAL_NUM_EXAMPLES=${RETRIEVAL_NUM_EXAMPLES:-512}
export RETRIEVAL_BATCH_SIZE=${RETRIEVAL_BATCH_SIZE:-128}
export NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-32}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-sherlock_sva_meg2sem_minilm_calibrated_diffusion}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-${RUN_TAG}_diffusion_eval512}

exec bash "$PROJECT_ROOT/submit-jobs/sherlock_sva_minilm_oracle_pipe_eval.sbatch"
