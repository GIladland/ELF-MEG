#!/usr/bin/env bash
#SBATCH --job-name=tgpt_diag
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/tang_gpt_checkpoint_diagnostics-%j.out
#SBATCH --error=submit-jobs/tang_gpt_checkpoint_diagnostics-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
export TMPDIR=${TMPDIR:-$DATA_ROOT/tmp}
export SKIP_ARC_SETUP=${SKIP_ARC_SETUP:-1}

export NPZ_PATH=${NPZ_PATH:-/data/engs-pnpl/glandau/elf-cache/tang_gpt/tang_gpt_seg10_text_pairs_train_storychunks8to18_t12_tanggpt.npz}
export CHECKPOINT=${CHECKPOINT:-/data/engs-pnpl/glandau/elf-runs/tang_gpt_storychunks_decoderprob90_continue_from_current_best_30ep_b32_a100_20260823/best.pt}
export TARGET_LATENTS_CACHE=${TARGET_LATENTS_CACHE:-/data/engs-pnpl/glandau/elf-cache/tang_gpt/tang_gpt_seg10_podcast_storychunks8to18_t12_tanggpt_elf_fullmask_k64_full_150ep_b32_short12h_a100_t5small_latents_fp16.npy}
export OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/tang_gpt_storychunks_checkpoint_diagnostics_20260823}

export INPUT_KEY=${INPUT_KEY:-input_embeddings}
export SENTENCE_KEY=${SENTENCE_KEY:-sentence}
export NUM_EXAMPLES=${NUM_EXAMPLES:-128}
export BATCH_SIZE=${BATCH_SIZE:-16}
export NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-32}
export CFG_SCALE=${CFG_SCALE:-1.0}
export SELF_COND_CFG_SCALE=${SELF_COND_CFG_SCALE:-1.0}
export SEED=${SEED:-7}

mkdir -p "$OUTPUT_DIR" "$TMPDIR"

if ! command -v conda >/dev/null 2>&1; then
  for conda_sh in \
    "${CONDA_SH:-}" \
    "/apps/system/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/data/engs-pnpl/glandau/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -n "$conda_sh" && -f "$conda_sh" ]]; then
      source "$conda_sh"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found; set CONDA_SH before launching." >&2
  exit 127
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_ROOT/xdg-cache}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${TORCH_HOME:-$DATA_ROOT/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/pip-cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$DATA_ROOT/conda-pkgs}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

cd "$PROJECT_ROOT"
ENV_STAMP_PATH="$TMP_ENV_ROOT/.elf_env_stamp"
if [[ "$SKIP_ARC_SETUP" == "1" && -d "$TMP_ENV_ROOT/bin" && -f "$ENV_STAMP_PATH" && "$(cat "$ENV_STAMP_PATH")" == "$ENV_STAMP" ]]; then
  echo "Skipping ARC env setup; reusing TMP_ENV_ROOT=$TMP_ENV_ROOT ENV_STAMP=$ENV_STAMP"
else
  PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" TMP_ENV_ROOT="$TMP_ENV_ROOT" ENV_STAMP="$ENV_STAMP" \
    bash "$PROJECT_ROOT/scripts/arc_setup_elf_env.sh"
fi
conda activate "$TMP_ENV_ROOT"

echo "NPZ_PATH=$NPZ_PATH"
echo "CHECKPOINT=$CHECKPOINT"
echo "TARGET_LATENTS_CACHE=$TARGET_LATENTS_CACHE"
echo "OUTPUT_DIR=$OUTPUT_DIR"

python scripts/eval_npz_semantic_checkpoint_diagnostics.py \
  --npz-path "$NPZ_PATH" \
  --checkpoint "$CHECKPOINT" \
  --target-latents-cache "$TARGET_LATENTS_CACHE" \
  --output-dir "$OUTPUT_DIR" \
  --input-key "$INPUT_KEY" \
  --sentence-key "$SENTENCE_KEY" \
  --num-examples "$NUM_EXAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --num-sampling-steps "$NUM_SAMPLING_STEPS" \
  --cfg-scale "$CFG_SCALE" \
  --self-cond-cfg-scale "$SELF_COND_CFG_SCALE" \
  --seed "$SEED" \
  --device cuda
