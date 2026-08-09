#!/usr/bin/env bash
#SBATCH --job-name=sh_ext_valnbr
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=190G
#SBATCH --output=submit-jobs/prepare_sherlock_sva_external_valneighbor_minilm_seed7-%j.out
#SBATCH --error=submit-jobs/prepare_sherlock_sva_external_valneighbor_minilm_seed7-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
SUITE_DIR=${SUITE_DIR:-$DATA_ROOT/sherlock_sva/coverage_suite_v1}
TRAIN=$DATA_ROOT/sherlock_sva/sherlock1to9_sva_no_coord_decl5to18_train_minilm_seed7.npz
VAL=$DATA_ROOT/sherlock_sva/sherlock12_sva_no_coord_decl5to18_val_minilm_seed7.npz
GUTENBERG=$DATA_ROOT/gutenberg/gutenberg_catalog_en_short5to18_simple_sv_no_coord_minilm_seed7.npz
CONVERSATION=$DATA_ROOT/conversation/conversation_clean_simple_sv_no_coord_candidates_minilm_seed7.npz
SELECTED=$SUITE_DIR/external_valneighbor_selected20k_minilm_seed7.npz
NEIGH=$SUITE_DIR/external_valneighbor_selected20k_gramneigh8x_minilm_seed7.npz
GATE_F1=${GATE_F1:-0.42}

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

if [[ ! -f "$SELECTED" ]]; then
  python scripts/select_semantic_npz_by_coverage.py \
    --candidate-npz "$GUTENBERG" \
    --candidate-name gutenberg_clean_sva \
    --candidate-npz "$CONVERSATION" \
    --candidate-name conversation_sva \
    --query-npz "$VAL" \
    --blocked-npz "$TRAIN" \
    --blocked-npz "$VAL" \
    --exclude-candidate-row-regex 'Doyle, Arthur Conan|Conan Doyle|Sherlock' \
    --output "$SELECTED" \
    --count 20000 \
    --per-query-candidates 256 \
    --per-query-lexical-candidates 512 \
    --lexical-token-max-candidates 30000 \
    --candidate-batch-size 4096 \
    --cosine-weight 0.35 \
    --lexical-weight 0.45 \
    --length-weight 0.10 \
    --order-weight 0.10 \
    --lexical-mode word_f1 \
    --max-per-query 256 \
    --seed 7 \
    --device cuda
fi

if [[ ! -f "$NEIGH" ]]; then
  python scripts/augment_sherlock_oracle_grammatical_neighborhood.py \
    --source-npz "$SELECTED" \
    --blocked-npz "$TRAIN" \
    --blocked-npz "$VAL" \
    --output "$NEIGH" \
    --variants-per-source 8 \
    --batch-size 512 \
    --device cuda \
    --local-files-only
fi

build_and_score() {
  local tag=$1
  shift
  local combined=$SUITE_DIR/${tag}_to_sherlock12_combined_minilm_seed7.npz
  python scripts/concat_semantic_npzs.py \
    "$@" \
    --val-npz "$VAL" \
    --output "$combined" \
    --dedupe-train \
    --dedupe-val \
    --drop-train-overlap-with-val
  python scripts/analyze_npz_coverage.py \
    --npz "$combined" \
    --name "$tag" \
    --val-num-examples 75 \
    --train-query-sample-size 1024 \
    --batch-size 64 \
    --seed 7 \
    --device cuda \
    --output-json "$SUITE_DIR/${tag}_coverage.json"
}

build_and_score external_valneighbor20k \
  --train-npz "$TRAIN" --train-limit 0 \
  --train-npz "$SELECTED" --train-limit 20000

build_and_score external_valneighbor20k_plus_neigh8x \
  --train-npz "$TRAIN" --train-limit 0 \
  --train-npz "$SELECTED" --train-limit 20000 \
  --train-npz "$NEIGH" --train-limit 160000

build_and_score external_valneighbor20k_plus_neigh8x_cleanbalanced50k \
  --train-npz "$TRAIN" --train-limit 0 \
  --train-npz "$SELECTED" --train-limit 20000 \
  --train-npz "$NEIGH" --train-limit 160000 \
  --train-npz "$SUITE_DIR/clean_balanced_selected400k_minilm_seed7.npz" --train-limit 50000

python - "$SUITE_DIR" "$GATE_F1" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

suite = Path(sys.argv[1])
gate = float(sys.argv[2])
names = [
    "external_valneighbor20k",
    "external_valneighbor20k_plus_neigh8x",
    "external_valneighbor20k_plus_neigh8x_cleanbalanced50k",
]
rows = []
for name in names:
    path = suite / f"{name}_coverage.json"
    data = json.loads(path.read_text())["results"][0]
    f1 = float(data["val_to_train"]["word_f1"]["mean"])
    recall = float(data["val_to_train"]["word_recall"]["mean"])
    cosine = float(data["val_to_train"]["cosine"]["mean"])
    rows.append((f1, recall, cosine, data["train_n"], name, data["npz"]))
rows.sort(reverse=True)
for f1, recall, cosine, train_n, name, npz in rows:
    print(f"coverage name={name} f1={f1:.6f} recall={recall:.6f} cosine={cosine:.6f} train_n={train_n} npz={npz}", flush=True)

best_f1, best_recall, best_cosine, train_n, best_name, best_npz = rows[0]
if best_f1 < gate:
    print(f"best coverage {best_f1:.6f} is below gate {gate:.6f}; not submitting ELF training", flush=True)
    sys.exit(0)

epochs = "70" if train_n <= 80000 else "30"
run_name = f"sherlock_sva_{best_name}_minilm_k64_epoch{epochs}_fullmask_long2d"
notes = (
    "Diagnostic, target-neighborhood selected run. Uses Sherlock12 held-out MiniLM/text only to select "
    "nearby non-Sherlock external Gutenberg/conversation candidates; exact held-out and Sherlock train text "
    "are blocked. This is not a clean generalization run; it tests whether non-exact local coverage is sufficient."
)
env = os.environ.copy()
env.update(
    {
        "RUN_NAME": run_name,
        "NPZ_PATH": best_npz,
        "EPOCHS": epochs,
        "WANDB_RUN_GROUP": "sherlock_target_neighbor_diagnostics",
        "WANDB_NOTES_TEXT": notes,
    }
)
result = subprocess.run(
    ["sbatch", "submit-jobs/sherlock_sva_coverage_gate_train_minilm_k64.sbatch"],
    check=True,
    text=True,
    capture_output=True,
    env=env,
)
print(result.stdout.strip(), flush=True)
PY
