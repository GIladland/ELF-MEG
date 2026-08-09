#!/usr/bin/env bash
#SBATCH --job-name=sh_sdev_cov
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=submit-jobs/prepare_sherlock_sva_sessiondev_aug_coverage-%j.out
#SBATCH --error=submit-jobs/prepare_sherlock_sva_sessiondev_aug_coverage-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
SUITE_DIR=${SUITE_DIR:-$DATA_ROOT/sherlock_sva/coverage_suite_v1/sessiondev_v1}
BASE=$DATA_ROOT/sherlock_sva/sherlock1to9_sva_no_coord_decl5to18_train_minilm_seed7.npz
DEV_ROW_RE='(?=.*"corpus": "Sherlock1")(?=.*"session": "(8|9|10)")'
SUPPORT=$SUITE_DIR/sherlock1to9_support_excl_sherlock1_ses8to10_minilm_seed7.npz
DEV=$SUITE_DIR/sherlock1_ses8to10_dev_minilm_seed7.npz

if ! command -v conda >/dev/null 2>&1; then
  for conda_sh in \
    "${CONDA_SH:-}" \
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
  echo "conda command not found; set CONDA_SH to the conda.sh path before launching." >&2
  exit 127
fi
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

if [[ ! -f "$SUPPORT" ]]; then
  python scripts/filter_semantic_npz_rows.py \
    --input "$BASE" \
    --output "$SUPPORT" \
    --exclude-row-regex "$DEV_ROW_RE"
fi
if [[ ! -f "$DEV" ]]; then
  python scripts/filter_semantic_npz_rows.py \
    --input "$BASE" \
    --output "$DEV" \
    --include-row-regex "$DEV_ROW_RE"
fi

make_recomb() {
  local mode=$1
  local output=$2
  if [[ -f "$output" ]]; then
    echo "Reusing $output"
    return
  fi
  python scripts/augment_sherlock_sva_recombinational.py \
    --train-npz "$SUPPORT" \
    --blocked-npz "$DEV" \
    --output "$output" \
    --mode "$mode" \
    --target-count 50000 \
    --candidate-count 160000 \
    --dev-count 512 \
    --selection-top-k 128 \
    --max-per-dev-query 128 \
    --batch-size 512 \
    --device cuda \
    --local-files-only
}

SLOT_AUG=$SUITE_DIR/support_recomb_slot50k_minilm_seed7.npz
PHRASE_AUG=$SUITE_DIR/support_recomb_phrasebank50k_minilm_seed7.npz
DEVCOVER_AUG=$SUITE_DIR/support_recomb_devcover50k_minilm_seed7.npz
HIGHFREQ_AUG=$SUITE_DIR/support_highfreq_local50k_cov95_minfrac70_minilm_seed7.npz

make_recomb slot "$SLOT_AUG"
make_recomb phrasebank "$PHRASE_AUG"
make_recomb devcover "$DEVCOVER_AUG"

if [[ ! -f "$HIGHFREQ_AUG" ]]; then
  python scripts/generate_highfreq_sherlock_augmentation.py \
    --train-npz "$SUPPORT" \
    --blocked-npz "$DEV" \
    --output "$HIGHFREQ_AUG" \
    --count 50000 \
    --token-coverage 0.95 \
    --min-highfreq-content-frac 0.70 \
    --chunk-mode local \
    --batch-size 512 \
    --device cuda \
    --local-files-only
fi

build_and_score() {
  local tag=$1
  shift
  local combined=$SUITE_DIR/${tag}_to_sessiondev_combined_minilm_seed7.npz
  python scripts/concat_semantic_npzs.py \
    "$@" \
    --val-npz "$DEV" \
    --output "$combined" \
    --dedupe-train \
    --dedupe-val \
    --drop-train-overlap-with-val
  python scripts/analyze_npz_coverage.py \
    --npz "$combined" \
    --name "$tag" \
    --val-num-examples 0 \
    --train-query-sample-size 1024 \
    --batch-size 64 \
    --seed 7 \
    --device cuda \
    --output-json "$SUITE_DIR/${tag}_coverage.json"
}

build_and_score support_only \
  --train-npz "$SUPPORT" --train-limit 0
build_and_score support_plus_slot50k \
  --train-npz "$SUPPORT" --train-limit 0 \
  --train-npz "$SLOT_AUG" --train-limit 0
build_and_score support_plus_phrase50k \
  --train-npz "$SUPPORT" --train-limit 0 \
  --train-npz "$PHRASE_AUG" --train-limit 0
build_and_score support_plus_devcover50k \
  --train-npz "$SUPPORT" --train-limit 0 \
  --train-npz "$DEVCOVER_AUG" --train-limit 0
build_and_score support_plus_highfreq50k \
  --train-npz "$SUPPORT" --train-limit 0 \
  --train-npz "$HIGHFREQ_AUG" --train-limit 0
build_and_score support_plus_all200k \
  --train-npz "$SUPPORT" --train-limit 0 \
  --train-npz "$SLOT_AUG" --train-limit 0 \
  --train-npz "$PHRASE_AUG" --train-limit 0 \
  --train-npz "$DEVCOVER_AUG" --train-limit 0 \
  --train-npz "$HIGHFREQ_AUG" --train-limit 0

python - <<'PY'
import json
from pathlib import Path
root = Path("/data/engs-pnpl/glandau/elf-cache/sherlock_sva/coverage_suite_v1/sessiondev_v1")
rows = []
for p in sorted(root.glob("*_coverage.json")):
    obj = json.loads(p.read_text())
    result = obj["results"][0]
    f1 = result["val_to_train"]["word_f1"]
    recall = result["val_to_train"]["word_recall"]
    cosine = result["val_to_train"]["cosine"]
    rows.append((f1["mean"], result["name"], result["train_n"], f1, recall, cosine))
rows.sort(reverse=True)
print("sessiondev coverage summary")
for _score, name, train_n, f1, recall, cosine in rows:
    print(
        f"{name} train_n={train_n} "
        f"f1_mean={f1['mean']:.3f} f1_med={f1['median']:.3f} "
        f"f1_ge50={f1['ge_0.5']:.3f} f1_ge70={f1['ge_0.7']:.3f} "
        f"recall={recall['mean']:.3f} cos={cosine['mean']:.3f}"
    )
PY
