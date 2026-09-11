#!/usr/bin/env bash
#SBATCH --job-name=prep_sem_variants
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=submit-jobs/prepare_middle10_semantic_variants-%j.out
#SBATCH --error=submit-jobs/prepare_middle10_semantic_variants-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
SOURCE_DIR=${SOURCE_DIR:-$DATA_ROOT/tang_gpt/oracle3072_middle10content_clockfix}
SOURCE_PREFIX=${SOURCE_PREFIX:-uts01_tang_gpt_lanczos_seg10_oracle3072_middle10content_clockfix}
MINILM_DIR=${MINILM_DIR:-$DATA_ROOT/tang_gpt/oracle3072_middle10content_clockfix_minilm}
AVG768_DIR=${AVG768_DIR:-$DATA_ROOT/tang_gpt/oracle3072_middle10content_clockfix_tangavg768}
MINILM_MODEL=${MINILM_MODEL:-sentence-transformers/all-MiniLM-L6-v2}
MINILM_MAX_LENGTH=${MINILM_MAX_LENGTH:-64}
PREDICTED_SOURCE_NPZ=${PREDICTED_SOURCE_NPZ:-$SOURCE_DIR/uts01_lanczos_seg10_test_pred3072_middle10content_clockfix.npz}
PREDICTED_AVG_OUTPUT=${PREDICTED_AVG_OUTPUT:-$AVG768_DIR/uts01_lanczos_seg10_test_pred3072_middle10content_clockfix_tangavg768.npz}

export PROJECT_ROOT DATA_ROOT
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export TORCH_PIP_SPEC=${TORCH_PIP_SPEC:-torch==2.7.1}
export TORCH_PIP_INDEX_URL=${TORCH_PIP_INDEX_URL:-https://download.pytorch.org/whl/cu118}

# arc_setup_elf_env.sh may run in a subprocess when the node-local environment
# is missing.  Cache exports made by that subprocess do not propagate back to
# this job, so define the shared caches in the calling shell as well.  This
# avoids Hugging Face falling back to the quota-limited home directory.
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$DATA_ROOT/xdg-cache}
export HF_HOME=${HF_HOME:-$DATA_ROOT/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export TORCH_HOME=${TORCH_HOME:-$DATA_ROOT/torch}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$DATA_ROOT/pip-cache}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$DATA_ROOT/conda-pkgs}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
mkdir -p \
  "$XDG_CACHE_HOME" \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TORCH_HOME" \
  "$PIP_CACHE_DIR" \
  "$CONDA_PKGS_DIRS"

source "$(conda info --base)/etc/profile.d/conda.sh"
stamp_path=$TMP_ENV_ROOT/.elf_env_stamp
if [[ ! -d "$TMP_ENV_ROOT/bin" || ! -f "$stamp_path" || "$(cat "$stamp_path")" != "$ENV_STAMP" ]]; then
  bash "$PROJECT_ROOT/scripts/arc_setup_elf_env.sh"
fi
conda activate "$TMP_ENV_ROOT"
cd "$PROJECT_ROOT"
mkdir -p "$MINILM_DIR" "$AVG768_DIR"

for split in train_val test; do
  source_npz=$SOURCE_DIR/${SOURCE_PREFIX}_${split}.npz
  python scripts/reembed_semantic_npz_sentences.py \
    --input-npz "$source_npz" \
    --output "$MINILM_DIR/${SOURCE_PREFIX}_${split}_minilm.npz" \
    --embedding-model-name "$MINILM_MODEL" \
    --pooling mean \
    --max-length "$MINILM_MAX_LENGTH" \
    --batch-size 256 \
    --device cuda

  python scripts/transform_semantic_npz.py \
    --input-npz "$source_npz" \
    --output "$AVG768_DIR/${SOURCE_PREFIX}_${split}_tangavg768.npz" \
    --operation mean-blocks \
    --block-count 4
done

python scripts/transform_semantic_npz.py \
  --input-npz "$PREDICTED_SOURCE_NPZ" \
  --output "$PREDICTED_AVG_OUTPUT" \
  --operation mean-blocks \
  --block-count 4

echo "Prepared matched semantic variants:"
ls -lh "$MINILM_DIR"/*.npz "$AVG768_DIR"/*.npz
