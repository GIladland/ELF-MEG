#!/usr/bin/env bash
#SBATCH --job-name=prep_fmri_ada
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=0-06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=submit-jobs/prepare_fmri_knowntext_ada002-%j.out
#SBATCH --error=submit-jobs/prepare_fmri_knowntext_ada002-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
MINILM_DIR=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
ADA_DIR=${ADA_DIR:-$DATA_ROOT/ada/fmri_knowntext_v1}
SOURCE_ALL=$MINILM_DIR/oracle_exact_minilm384_all12098_testfirst.npz
SOURCE_TRAINVAL=$MINILM_DIR/oracle_exact_minilm384_train_val.npz
ADA_ALL=$ADA_DIR/oracle_exact_ada002_all12098_testfirst.npz
ADA_TRAINVAL=$ADA_DIR/oracle_exact_ada002_train_val.npz
ADA_MODEL=${ADA_MODEL:-text-embedding-ada-002}
ADA_BATCH_SIZE=${ADA_BATCH_SIZE:-128}
PYTHON_BIN=${PYTHON_BIN:-/data/engs-pnpl/glandau/BrainDiffusion/elf-env/bin/python}
OPENAI_API_KEY_FILE=${OPENAI_API_KEY_FILE:-/data/engs-pnpl/glandau/BrainDiffusion/secrets/openai_api_key}

mkdir -p "$ADA_DIR" "$DATA_ROOT/tmp"
cd "$PROJECT_ROOT"

if [[ -z "${OPENAI_API_KEY:-}" && -f "$OPENAI_API_KEY_FILE" ]]; then
  export OPENAI_API_KEY
  OPENAI_API_KEY="$(tr -d '\r\n' < "$OPENAI_API_KEY_FILE")"
fi
if [[ ! -f "$ADA_ALL" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Missing OpenAI credential for ADA preparation." >&2
  exit 2
fi

"$PYTHON_BIN" scripts/reembed_semantic_npz_openai.py \
  --input-npz "$SOURCE_ALL" \
  --output "$ADA_ALL" \
  --model "$ADA_MODEL" \
  --batch-size "$ADA_BATCH_SIZE"

"$PYTHON_BIN" - "$ADA_ALL" "$SOURCE_TRAINVAL" "$ADA_TRAINVAL" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

ada_all_path, source_trainval_path, output_path = map(Path, sys.argv[1:])
if output_path.exists():
    with np.load(output_path, allow_pickle=True) as existing, np.load(
        source_trainval_path, allow_pickle=True
    ) as source:
        existing_embeddings = np.asarray(existing["input_embeddings"])
        expected_sentences = np.asarray(source["sentence"], dtype=object)
        if existing_embeddings.shape != (len(expected_sentences), 1536):
            raise ValueError(
                f"Existing ADA train/validation shape is invalid: {existing_embeddings.shape}."
            )
        if not np.array_equal(
            np.asarray(existing["sentence"], dtype=object), expected_sentences
        ):
            raise ValueError("Existing ADA train/validation sentences are misaligned.")
        split_values, split_counts = np.unique(existing["split"], return_counts=True)
        summary = {
            "output": str(output_path),
            "embedding_shape": list(existing_embeddings.shape),
            "embedding_model_name": "text-embedding-ada-002",
            "split_counts": {
                str(split): int(count) for split, count in zip(split_values, split_counts)
            },
            "validated_existing_output": True,
        }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    raise SystemExit(0)

with np.load(ada_all_path, allow_pickle=True) as ada_all, np.load(
    source_trainval_path, allow_pickle=True
) as trainval:
    embeddings = np.asarray(ada_all["input_embeddings"], dtype=np.float32)
    trainval_sentences = np.asarray(trainval["sentence"], dtype=object)
    offset = len(embeddings) - len(trainval_sentences)
    if offset < 0 or not np.array_equal(
        np.asarray(ada_all["sentence"][offset:], dtype=object),
        trainval_sentences,
    ):
        raise ValueError(
            "ADA all-corpus suffix does not exactly match the train/validation sentence order."
        )
    arrays = {key: trainval[key] for key in trainval.files if key != "input_embeddings"}
    arrays["input_embeddings"] = embeddings[offset:]
    schema = {
        "source_ada_all": str(ada_all_path),
        "source_trainval": str(source_trainval_path),
        "embedding_model_name": "text-embedding-ada-002",
        "embedding_shape": list(arrays["input_embeddings"].shape),
        "normalized": True,
        "alignment_rule": f"validated all-corpus suffix offset {offset} by exact sentence equality",
    }
    arrays["schema_json"] = np.asarray(json.dumps(schema))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    output_path.with_suffix(".summary.json").write_text(json.dumps(schema, indent=2))
    print(json.dumps(schema, indent=2))
PY

echo "Prepared ADA corpora: $ADA_ALL and $ADA_TRAINVAL"
