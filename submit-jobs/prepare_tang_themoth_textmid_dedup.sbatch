#!/usr/bin/env bash
#SBATCH --job-name=prep_tm_textmid_dedup
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/prepare_tang_themoth_textmid_dedup-%j.out
#SBATCH --error=submit-jobs/prepare_tang_themoth_textmid_dedup-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
MEG2SEM_ROOT=${MEG2SEM_ROOT:-/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM}
PYTHON=${MEG2SEM_PYTHON:-/data/engs-pnpl/glandau/conda/envs/meg2sem/bin/python}
ROOT=${OUTPUT_ROOT:-/data/engs-pnpl/glandau/elf-cache/tang_themoth_apples_exact384_textmid_seg10_v3}
PACKED_ROOT=${PACKED_ROOT:-$ROOT/packed}
PREFIX=${PREFIX:-tang_themoth_exact384_minilm384_textmid_seg10000}
ORACLE_TEST=${ORACLE_TEST:-/data/engs-pnpl/glandau/elf-cache/minilm/knowntext_unseenbrain_v1/oracle_exact_minilm384_test107.npz}

source_train=$PACKED_ROOT/${PREFIX}_train_fp16.npz
dedup_train=$PACKED_ROOT/${PREFIX}_train_dedup_fp16.npz
val_npz=$PACKED_ROOT/${PREFIX}_val_fp16.npz
test_npz=$PACKED_ROOT/${PREFIX}_test_fp16.npz

cd "$PROJECT_ROOT"
"$PYTHON" scripts/deduplicate_packed_meg2sem.py \
  --input "$source_train" \
  --output "$dedup_train" \
  --summary-json "$ROOT/train_dedup_summary.json" \
  --embedding-key embeddings_minilm \
  --overwrite

"$PYTHON" "$MEG2SEM_ROOT/utils/check_packed_meg2sem_health.py" \
  --npz "train=$dedup_train" \
  --npz "val=$val_npz" \
  --npz "test=$test_npz" \
  --embedding-type MINILM \
  --expect-embedding-dim 384 \
  --expect-segment-ms 10000 \
  --max-meg-rows 512 \
  --meg-sample-mode random \
  --out-json "$ROOT/packed_health_dedup.json"

"$PYTHON" scripts/validate_tang_themoth_exact384_contract.py \
  --oracle-test-npz "$ORACLE_TEST" \
  --meg-test-npz "$test_npz" \
  --story birthofanation \
  --output-json "$ROOT/exact384_cross_modal_contract.json"

"$PYTHON" scripts/compute_semantic_train_mean.py \
  --input-npz "$dedup_train" \
  --output-npz "$ROOT/tang_themoth_exact384_minilm384_train_mean.npz" \
  --embedding-key embeddings_minilm

echo "Prepared deduplicated phrase-centered train pack: $dedup_train"
