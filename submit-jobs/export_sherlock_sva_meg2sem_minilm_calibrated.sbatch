#!/usr/bin/env bash
#SBATCH --job-name=sva_mlm_map
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_cc:7.0|gpu_cc:8.0|gpu_cc:8.9|gpu_cc:9.0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/export_sherlock_sva_meg2sem_minilm_calibrated-%j.out
#SBATCH --error=submit-jobs/export_sherlock_sva_meg2sem_minilm_calibrated-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
NPZ_PATH=${NPZ_PATH:-$DATA_ROOT/sherlock_sva/meg2sem/sherlock1to9_sva_no_coord_decl5to18_train_minilm_meg2sem_segment3000_fp16.npz}
MEG2SEM_CHECKPOINT=${MEG2SEM_CHECKPOINT:-/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/dt8z7o55/checkpoints/sherlock_sva_minilm_books1to9_to_sherlock12_bs60_seed49-epoch=852-val_loss=26.45.ckpt}
RUN_TAG=${RUN_TAG:-sherlock_sva_meg2sem_dt8z7o55_minilm_traincal_val512_test512}
OUTPUT_DIR=${OUTPUT_DIR:-$DATA_ROOT/sherlock_sva/meg2sem/calibrated/$RUN_TAG}
BATCH_SIZE=${BATCH_SIZE:-32}

if [[ ! -x "$TMP_ENV_ROOT/bin/python" || ! -f "$TMP_ENV_ROOT/.elf_env_stamp" || "$(<"$TMP_ENV_ROOT/.elf_env_stamp")" != "$ENV_STAMP" ]]; then
  echo "Compatible ELF environment is missing or has the wrong stamp: $TMP_ENV_ROOT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$TMP_ENV_ROOT/bin/python" scripts/export_calibrated_meg2sem_minilm.py \
  --npz "$NPZ_PATH" \
  --checkpoint "$MEG2SEM_CHECKPOINT" \
  --output-npz "$OUTPUT_DIR/mapped_minilm384.npz" \
  --metrics-json "$OUTPUT_DIR/metrics.json" \
  --validation-examples 512 \
  --test-examples 512 \
  --batch-size "$BATCH_SIZE" \
  --device cuda
