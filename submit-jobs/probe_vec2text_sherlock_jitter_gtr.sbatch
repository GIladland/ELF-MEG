#!/usr/bin/env bash
#SBATCH --job-name=sh_v2t_probe
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=priority
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=submit-jobs/probe_vec2text_sherlock_jitter_gtr-%j.out
#SBATCH --error=submit-jobs/probe_vec2text_sherlock_jitter_gtr-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
OUTPUT_DIR=${OUTPUT_DIR:-$DATA_ROOT/sherlock_sva/vec2text_jitter_probe_gtr_sherlock12_5x20_seed7}

HELDOUT_NPZ=${HELDOUT_NPZ:-$DATA_ROOT/sherlock_sva/sherlock12_sva_no_coord_decl5to18_val_minilm_seed7.npz}
GEOMETRY_NPZ=${GEOMETRY_NPZ:-$DATA_ROOT/sherlock/sherlock_books1to9_sentences_minilm_seed7.npz}
BASELINE_NPZ=${BASELINE_NPZ:-$DATA_ROOT/sherlock_sva/sherlock1to9_sva_no_coord_decl5to18_train_minilm_seed7.npz}

source "$(conda info --base)/etc/profile.d/conda.sh"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$DATA_ROOT/xdg-cache}
export HF_HOME=${HF_HOME:-$DATA_ROOT/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export TORCH_HOME=${TORCH_HOME:-$DATA_ROOT/torch}
export WANDB_DIR=${WANDB_DIR:-$DATA_ROOT/wandb}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$DATA_ROOT/triton-cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$DATA_ROOT/pip-cache}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$DATA_ROOT/conda-pkgs}
export TMPDIR=${TMPDIR:-$DATA_ROOT/tmp}
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

mkdir -p "$OUTPUT_DIR" "$TMPDIR"
cd "$PROJECT_ROOT"

ENV_STAMP_PATH="$TMP_ENV_ROOT/.elf_env_stamp"
if [[ "${SKIP_ARC_SETUP:-1}" == "1" && -d "$TMP_ENV_ROOT/bin" && -f "$ENV_STAMP_PATH" && "$(cat "$ENV_STAMP_PATH")" == "$ENV_STAMP" ]]; then
  echo "Reusing stamped ARC env at $TMP_ENV_ROOT"
else
  PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" TMP_ENV_ROOT="$TMP_ENV_ROOT" ENV_STAMP="$ENV_STAMP" \
    bash "$PROJECT_ROOT/scripts/arc_setup_elf_env.sh"
fi
conda activate "$TMP_ENV_ROOT"

python - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("vec2text") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-user", "vec2text"])
PY

python scripts/probe_vec2text_sherlock_jitter.py \
  --heldout-npz "$HELDOUT_NPZ" \
  --geometry-npz "$GEOMETRY_NPZ" \
  --baseline-npz "$BASELINE_NPZ" \
  --output-dir "$OUTPUT_DIR" \
  --num-heldout "${NUM_HELDOUT:-5}" \
  --heldout-start "${HELDOUT_START:-0}" \
  --samples-per-scale "${SAMPLES_PER_SCALE:-20}" \
  --jitter-angles "${JITTER_ANGLES:-0,0.02,0.04,0.08,0.14}" \
  --neighbor-k "${NEIGHBOR_K:-48}" \
  --vec2text-steps "${VEC2TEXT_STEPS:-8}" \
  --sequence-beam-width "${SEQUENCE_BEAM_WIDTH:-0}" \
  --vec2text-batch-size "${VEC2TEXT_BATCH_SIZE:-16}" \
  --gtr-batch-size "${GTR_BATCH_SIZE:-64}" \
  --minilm-batch-size "${MINILM_BATCH_SIZE:-128}" \
  --seed "${SEED:-7}" \
  --device cuda
