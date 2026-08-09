#!/usr/bin/env bash
#SBATCH --job-name=sh_v2t_proto
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=priority
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --output=submit-jobs/prepare_sherlock_sva_vec2text_prototype_gtr-%j.out
#SBATCH --error=submit-jobs/prepare_sherlock_sva_vec2text_prototype_gtr-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
SUITE_DIR=${SUITE_DIR:-$DATA_ROOT/sherlock_sva/coverage_suite_v1}

TRAIN=${TRAIN:-$DATA_ROOT/sherlock_sva/sherlock1to9_sva_no_coord_decl5to18_train_minilm_seed7.npz}
VAL=${VAL:-$DATA_ROOT/sherlock_sva/sherlock12_sva_no_coord_decl5to18_val_minilm_seed7.npz}
PROTOTYPES=${PROTOTYPES:-$DATA_ROOT/sherlock/sherlock_books1to9_sentences_minilm_seed7.npz}

RUN_TAG=${RUN_TAG:-vec2text_gtr_trainprototype_sherlock1to9_50k}
AUG_NPZ=${AUG_NPZ:-$SUITE_DIR/${RUN_TAG}_minilm_seed7.npz}
INNER_DEV_NPZ=${INNER_DEV_NPZ:-$SUITE_DIR/${RUN_TAG}_innerdev_minilm_seed7.npz}
SELECTED_NPZ=${SELECTED_NPZ:-$SUITE_DIR/${RUN_TAG}_selected_minilm_seed7.npz}

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

python - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("vec2text") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-user", "vec2text"])
PY

python scripts/generate_vec2text_train_prototype_aug.py \
  --prototype-npz "$PROTOTYPES" \
  --base-train-npz "$TRAIN" \
  --blocked-npz "$VAL" \
  --output "$AUG_NPZ" \
  --inner-dev-output "$INNER_DEV_NPZ" \
  --target-count "${TARGET_COUNT:-50000}" \
  --raw-multiplier "${RAW_MULTIPLIER:-1.8}" \
  --prototype-count "${PROTOTYPE_COUNT:-3500}" \
  --jitter-angles "${JITTER_ANGLES:-0.04,0.08,0.10}" \
  --neighbor-k "${NEIGHBOR_K:-48}" \
  --density-k "${DENSITY_K:-32}" \
  --vec2text-steps "${VEC2TEXT_STEPS:-6}" \
  --vec2text-batch-size "${VEC2TEXT_BATCH_SIZE:-16}" \
  --gtr-batch-size "${GTR_BATCH_SIZE:-96}" \
  --minilm-batch-size "${MINILM_BATCH_SIZE:-256}" \
  --filter-mode "${FILTER_MODE:-simple_sv_no_coord}" \
  --min-source-word-recall "${MIN_SOURCE_WORD_RECALL:-0.20}" \
  --min-source-minilm-cosine "${MIN_SOURCE_MINILM_COSINE:-0.60}" \
  --max-source-minilm-cosine "${MAX_SOURCE_MINILM_COSINE:-0.995}" \
  --seed "${SEED:-7}" \
  --device cuda

python scripts/select_semantic_npz_by_coverage.py \
  --candidate-npz "$AUG_NPZ" \
  --candidate-name vec2text_train_prototype_gtr \
  --query-npz "$INNER_DEV_NPZ" \
  --blocked-npz "$TRAIN" \
  --blocked-npz "$VAL" \
  --output "$SELECTED_NPZ" \
  --count "${SELECT_COUNT:-50000}" \
  --per-query-candidates 96 \
  --per-query-lexical-candidates 192 \
  --lexical-token-max-candidates 20000 \
  --candidate-batch-size 2048 \
  --cosine-weight "${SELECT_COSINE_WEIGHT:-0.35}" \
  --lexical-weight "${SELECT_LEXICAL_WEIGHT:-0.45}" \
  --length-weight "${SELECT_LENGTH_WEIGHT:-0.10}" \
  --order-weight "${SELECT_ORDER_WEIGHT:-0.10}" \
  --lexical-mode idf_f1 \
  --max-per-query 96 \
  --seed "${SEED:-7}" \
  --device cuda

for BUDGET in ${BUDGETS:-10000 50000}; do
  TAG=${RUN_TAG}_selected${BUDGET}
  COMBINED=$SUITE_DIR/${TAG}_to_sherlock12_combined_minilm_seed7.npz
  COVERAGE_JSON=$SUITE_DIR/${TAG}_coverage.json
  python scripts/concat_semantic_npzs.py \
    --train-npz "$TRAIN" --train-limit 0 \
    --train-npz "$SELECTED_NPZ" --train-limit "$BUDGET" \
    --val-npz "$VAL" \
    --output "$COMBINED" \
    --dedupe-train \
    --dedupe-val \
    --drop-train-overlap-with-val
  python scripts/analyze_npz_coverage.py \
    --npz "$COMBINED" \
    --name "$TAG" \
    --val-num-examples 75 \
    --train-query-sample-size 1024 \
    --batch-size 64 \
    --seed "${SEED:-7}" \
    --device cuda \
    --output-json "$COVERAGE_JSON"

  SCORE=$(python - "$COVERAGE_JSON" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1]))
score = data["results"][0]["val_to_train"]["word_f1"]["mean"]
print(f"{score:.6f}")
PY
)
  echo "coverage_gate tag=$TAG val_to_train_word_f1=$SCORE threshold=${TRAIN_GATE_F1:-0.30}"
  if python - "$SCORE" "${TRAIN_GATE_F1:-0.30}" <<'PY'
import sys
sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
PY
  then
    TRAIN_EPOCHS=${TRAIN_EPOCHS:-80}
    TRAIN_RUN=${TAG}_minilm_k64_epoch${TRAIN_EPOCHS}_fullmask_long2d
    sbatch --clusters=htc \
      --export=ALL,RUN_NAME="$TRAIN_RUN",NPZ_PATH="$COMBINED",EPOCHS="$TRAIN_EPOCHS",TARGET_LATENTS_CACHE="$SUITE_DIR/${TAG}_t5small_latents_fp16.npy",WANDB_RUN_GROUP="sherlock_vec2text_prototype_nonleaky" \
      submit-jobs/sherlock_sva_coverage_gate_train_minilm_k64.sbatch
  else
    echo "coverage below gate; not launching train for $TAG"
  fi
done
