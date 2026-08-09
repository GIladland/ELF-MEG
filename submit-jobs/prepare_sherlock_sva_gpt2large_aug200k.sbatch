#!/usr/bin/env bash
#SBATCH --job-name=sh_gpt2_aug
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --output=submit-jobs/prepare_sherlock_sva_gpt2large_aug200k-%j.out
#SBATCH --error=submit-jobs/prepare_sherlock_sva_gpt2large_aug200k-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
SUITE_DIR=${SUITE_DIR:-$DATA_ROOT/sherlock_sva/coverage_suite_v1}
TRAIN=${TRAIN:-$DATA_ROOT/sherlock_sva/sherlock1to9_sva_no_coord_decl5to18_train_minilm_seed7.npz}
VAL=${VAL:-$DATA_ROOT/sherlock_sva/sherlock12_sva_no_coord_decl5to18_val_minilm_seed7.npz}
OUTPUT=${OUTPUT:-$SUITE_DIR/sherlock1to9_gpt2large_aug200k_minilm_seed7.npz}
LM_CHECKPOINT=${LM_CHECKPOINT:-$SUITE_DIR/gpt2large_sherlock1to9_checkpoint}
TRAIN_STEPS=${TRAIN_STEPS:-1500}
PROMPT_MODE=${PROMPT_MODE:-prefix}

source "$(conda info --base)/etc/profile.d/conda.sh"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$DATA_ROOT/xdg-cache}
export HF_HOME=${HF_HOME:-$DATA_ROOT/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export TMPDIR=${TMPDIR:-$DATA_ROOT/tmp}

mkdir -p "$SUITE_DIR" "$TMPDIR"
cd "$PROJECT_ROOT"
ENV_STAMP_PATH="$TMP_ENV_ROOT/.elf_env_stamp"
if [[ "${SKIP_ARC_SETUP:-1}" == "1" && -d "$TMP_ENV_ROOT/bin" && -f "$ENV_STAMP_PATH" && "$(cat "$ENV_STAMP_PATH")" == "$ENV_STAMP" ]]; then
  echo "Reusing stamped ARC env at $TMP_ENV_ROOT"
else
  PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" TMP_ENV_ROOT="$TMP_ENV_ROOT" ENV_STAMP="$ENV_STAMP" \
    bash "$PROJECT_ROOT/scripts/arc_setup_elf_env.sh"
fi
conda activate "$TMP_ENV_ROOT"

python scripts/generate_sherlock_lm_augmentation.py \
  --train-npz "$TRAIN" \
  --blocked-npz "$VAL" \
  --output "$OUTPUT" \
  --lm-checkpoint-dir "$LM_CHECKPOINT" \
  --lm-model-name gpt2-large \
  --target-count 200000 \
  --train-steps "$TRAIN_STEPS" \
  --train-batch-size 4 \
  --gradient-accumulation 8 \
  --prompt-mode "$PROMPT_MODE" \
  --sample-batch-size 128 \
  --embedding-batch-size 512 \
  --device cuda \
  --local-files-only
