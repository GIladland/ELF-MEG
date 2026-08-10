#!/usr/bin/env bash
#SBATCH --job-name=sh_ada_orc_fm
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=submit-jobs/sherlock_sva_plus_oraclevalaug64x_ada_k64_epoch300_fullmask_long2d-%j.out
#SBATCH --error=submit-jobs/sherlock_sva_plus_oraclevalaug64x_ada_k64_epoch300_fullmask_long2d-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
SHERLOCK_DIR=${SHERLOCK_DIR:-$DATA_ROOT/sherlock_sva}
ORACLE_TAG=${ORACLE_TAG:-sherlock1to9_sva_no_coord_decl5to18_plus_oraclevalaug64x_to_sherlock12_sva_no_coord_decl5to18}
NPZ_PATH=${NPZ_PATH:-$SHERLOCK_DIR/${ORACLE_TAG}_combined_ada002_seed7.npz}
OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/${ORACLE_TAG}_ada002_k64_epoch300_fullmask_long2d}
TARGET_LATENTS_CACHE=${TARGET_LATENTS_CACHE:-$SHERLOCK_DIR/${ORACLE_TAG}_combined_ada002_fullmask_t5small_latents_fp16.npy}
PYTHON_BIN=${PYTHON_BIN:-/data/engs-pnpl/glandau/BrainDiffusion/elf-env/bin/python}

export WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
export WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-conditioning_ada_sherlock_oracle}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-${ORACLE_TAG}_ada002_k64_epoch300_fullmask_long2d}
export WANDB_DIR=${WANDB_DIR:-$DATA_ROOT/wandb}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-$DATA_ROOT/wandb-data}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$DATA_ROOT/wandb-cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$DATA_ROOT/xdg-cache}
export HF_HOME=${HF_HOME:-$DATA_ROOT/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export TORCH_HOME=${TORCH_HOME:-$DATA_ROOT/torch}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$DATA_ROOT/pip-cache}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export TMPDIR=${TMPDIR:-$DATA_ROOT/tmp}

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"
cd "$PROJECT_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 127
fi
if [[ ! -f "$NPZ_PATH" ]]; then
  echo "Missing ADA NPZ: $NPZ_PATH" >&2
  echo "Run submit-jobs/prepare_sherlock_sva_oracle_valaug_ada_seed7.sbatch first." >&2
  exit 1
fi

VAL_N=${VAL_N:-$("$PYTHON_BIN" - "$NPZ_PATH" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1]).with_suffix(".summary.json")
if not summary_path.exists():
    raise SystemExit(f"Missing summary JSON for validation count: {summary_path}")
with summary_path.open("r", encoding="utf-8") as handle:
    val_examples = json.load(handle).get("val_examples")
if not isinstance(val_examples, int) or val_examples <= 0:
    raise SystemExit(f"Invalid val_examples in {summary_path}: {val_examples!r}")
print(val_examples)
PY
)}

echo "Training ADA oraclevalaug ELF baseline"
echo "NPZ_PATH=$NPZ_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "TARGET_LATENTS_CACHE=$TARGET_LATENTS_CACHE"
echo "VAL_N=$VAL_N"
echo "PYTHON_BIN=$PYTHON_BIN"

"$PYTHON_BIN" scripts/train_npz_semantic_to_elf.py \
  --npz-path "$NPZ_PATH" \
  --use_wandb \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_group "$WANDB_RUN_GROUP" \
  --wandb_run_name "$WANDB_RUN_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --device cuda \
  --num-examples 0 \
  --val-num-examples "$VAL_N" \
  --eval-num-examples "$VAL_N" \
  --retrieval-num-examples "$VAL_N" \
  --retrieval-eval-every 2000 \
  --retrieval-batch-size 256 \
  --epochs 300 \
  --batch_size 64 \
  --eval_every 1000 \
  --num_sampling_steps 32 \
  --train-timestep-mode sampling_schedule \
  --train-timestep-steps 32 \
  --context_length 64 \
  --semantic-hidden-dim 4096 \
  --last_n_blocks -1 \
  --lr 1e-4 \
  --cfg_scale 1.0 \
  --self_cond_cfg_scale 1.0 \
  --cond_dropout_prob 0.0 \
  --denoiser_loss_weight 1.0 \
  --decoder_loss_weight 1.0 \
  --train-target-mask-mode full \
  --target-latents-mode cache \
  --target-latents-cache "$TARGET_LATENTS_CACHE" \
  --target-latents-cache-dtype float16 \
  --target-latent-cache-chunk-size 4096 \
  --target-latent-encode-batch-size 128 \
  --generation-t5-retrieval \
  --wandb_sample_examples 16 \
  --save-best-checkpoint \
  --seed 7 \
  --wandb_notes "True text-embedding-ada-002 oracle diagnostic. Same Sherlock oraclevalaug train/75-val split and full target-mask ELF training recipe that gave MiniLM words_overlap about 0.73, with only the conditioning embedding swapped to ADA."
