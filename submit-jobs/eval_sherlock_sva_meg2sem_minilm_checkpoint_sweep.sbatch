#!/usr/bin/env bash
#SBATCH --job-name=sva_mlm_sem
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_cc:7.0|gpu_cc:8.0|gpu_cc:8.9|gpu_cc:9.0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/eval_sherlock_sva_meg2sem_minilm_checkpoint_sweep-%j.out
#SBATCH --error=submit-jobs/eval_sherlock_sva_meg2sem_minilm_checkpoint_sweep-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
NPZ_PATH=${NPZ_PATH:-$DATA_ROOT/sherlock_sva/meg2sem/sherlock1to9_sva_no_coord_decl5to18_train_minilm_meg2sem_segment3000_fp16.npz}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/dt8z7o55/checkpoints}
OUTPUT_JSON=${OUTPUT_JSON:-$DATA_ROOT/meg2sem-evals/dt8z7o55_minilm_semantic_ceiling_headtail512.json}
BATCH_SIZE=${BATCH_SIZE:-32}

if [[ ! -x "$TMP_ENV_ROOT/bin/python" || ! -f "$TMP_ENV_ROOT/.elf_env_stamp" || "$(<"$TMP_ENV_ROOT/.elf_env_stamp")" != "$ENV_STAMP" ]]; then
  echo "Compatible ELF environment is missing or has the wrong stamp: $TMP_ENV_ROOT" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$OUTPUT_JSON")"

checkpoints=(
  "$CHECKPOINT_ROOT/sherlock_sva_minilm_books1to9_to_sherlock12_bs60_seed49-epoch=49-val_loss=18.51.ckpt"
  "$CHECKPOINT_ROOT/sherlock_sva_minilm_books1to9_to_sherlock12_bs60_seed49-epoch=200-val_loss=24.38.ckpt"
  "$CHECKPOINT_ROOT/sherlock_sva_minilm_books1to9_to_sherlock12_bs60_seed49-epoch=400-val_loss=24.63.ckpt"
  "$CHECKPOINT_ROOT/sherlock_sva_minilm_books1to9_to_sherlock12_bs60_seed49-epoch=600-val_loss=25.38.ckpt"
  "$CHECKPOINT_ROOT/sherlock_sva_minilm_books1to9_to_sherlock12_bs60_seed49-epoch=852-val_loss=26.45.ckpt"
  "$CHECKPOINT_ROOT/sherlock_sva_minilm_books1to9_to_sherlock12_bs60_seed49-epoch=999-val_loss=27.19.ckpt"
)

args=(
  scripts/eval_meg2sem_semantic_ceiling.py
  --npz "$NPZ_PATH"
  --output "$OUTPUT_JSON"
  --head-examples 512
  --tail-examples 512
  --batch-size "$BATCH_SIZE"
  --device cuda
)
for checkpoint in "${checkpoints[@]}"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 2
  fi
  args+=(--checkpoint "$checkpoint")
done

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$TMP_ENV_ROOT/bin/python" "${args[@]}"
