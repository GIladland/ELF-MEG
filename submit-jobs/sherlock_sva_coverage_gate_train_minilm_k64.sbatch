#!/usr/bin/env bash
#SBATCH --job-name=sh_cov_train
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=submit-jobs/sherlock_sva_coverage_gate_train_minilm_k64-%j.out
#SBATCH --error=submit-jobs/sherlock_sva_coverage_gate_train_minilm_k64-%j.err

set -euo pipefail

RUN_NAME=${RUN_NAME:?RUN_NAME is required}
NPZ_PATH=${NPZ_PATH:?NPZ_PATH is required}
EPOCHS=${EPOCHS:?EPOCHS is required}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
TARGET_LATENTS_CACHE=${TARGET_LATENTS_CACHE:-$DATA_ROOT/sherlock_sva/coverage_suite_v1/${RUN_NAME}_t5small_latents_fp16.npy}

export NPZ_PATH
export OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/$RUN_NAME}
export TARGET_LATENTS_CACHE
export SKIP_ARC_SETUP=${SKIP_ARC_SETUP:-1}
export WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
export WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-sherlock_nonleaky_coverage_suite}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$RUN_NAME}
export WANDB_NOTES_TEXT=${WANDB_NOTES_TEXT:-Non-leaky MiniLM pooled K64 coverage-gated run. Selection used Sherlock1-9 only; Doyle/Sherlock Gutenberg rows and exact validation text were excluded.}
export WANDB_DIR=${WANDB_DIR:-$DATA_ROOT/wandb}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-$DATA_ROOT/wandb-data}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$DATA_ROOT/wandb-cache}
export TMPDIR=${TMPDIR:-$DATA_ROOT/tmp}

VAL_N=$(python - "$NPZ_PATH" <<'PY'
import json
import sys
from pathlib import Path
summary = Path(sys.argv[1]).with_suffix(".summary.json")
print(json.loads(summary.read_text())["val_examples"])
PY
)

bash ./scripts/arc_run_npz_semantic_to_elf.sh \
  --device cuda \
  --num-examples 0 \
  --val-num-examples "$VAL_N" \
  --eval-num-examples "$VAL_N" \
  --retrieval-num-examples "$VAL_N" \
  --retrieval-eval-every 2000 \
  --retrieval-batch-size 256 \
  --epochs "$EPOCHS" \
  --batch_size 64 \
  --eval_every 2000 \
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
  --wandb_notes "$WANDB_NOTES_TEXT"
