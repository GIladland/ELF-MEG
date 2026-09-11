#!/usr/bin/env bash
#SBATCH --job-name=qc4_oof5
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=gpu_cc:8.0|gpu_cc:8.9|gpu_cc:9.0
#SBATCH --array=0-4%5
#SBATCH --output=submit-jobs/train_qc4wyals_nested_oof_folds_20260908-%A_%a.out
#SBATCH --error=submit-jobs/train_qc4wyals_nested_oof_folds_20260908-%A_%a.err

set -euo pipefail

brain_root=/data/engs-pnpl/glandau/BrainDiffusion/ELF
meg_root=/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM
meg_python=/data/engs-pnpl/glandau/conda/envs/meg2sem/bin/python
source_root=/data/engs-pnpl/glandau/elf-cache/tang_themoth_apples_exact_ada002_textmid_seg10_v1
source_train=$source_root/packed/tang_themoth_exact384_ada002_textmid_seg10000_train_dedup_fp16.npz
oof_root=/data/engs-pnpl/glandau/elf-cache/tang_themoth_qc4wyals_nested_oof_20260908
aux=/data/engs-pnpl/glandau/elf-cache/tang_themoth_apples_semantic_neighbors_ada002_v1/tang_exact_trainanchored_sherlock_sva_lb2themoth_wc8to16_maxcos_top8192_aux_ada_fp16_uncompressed.npz
sherlock=/data/engs-pnpl/glandau/elf-cache/sherlock_sva/meg2sem/sherlock1to9_sva_no_coord_decl5to18_train_ada002_meg2sem_segment8000_fp16_uncompressed.npz
lb2_len=/data/engs-pnpl/glandau/elf-cache/libribrain2_relaxed/meg2sem_podcast_lenbalanced/lb2_themoth_short_content8to16_ses1to28_val29_test30_ada002_segment8000_train_fp16_uncompressed.npz
lb2_clean=/data/engs-pnpl/glandau/elf-cache/libribrain2_relaxed/meg2sem_podcast_clean/lb2_themoth_short_content5to30_ses1to28_val29_test30_ada002_segment8000_train_fp16.npz
lb2_mixed=/data/engs-pnpl/glandau/elf-cache/libribrain2_relaxed/meg2sem/lb2_timit_mochatimit_themoth_short_content5to30_train_ada002_meg2sem_segment8000_fp16.npz
banks=$sherlock:$lb2_len:$lb2_clean:$lb2_mixed

fold=$SLURM_ARRAY_TASK_ID
seed=$((820 + fold))
run_name=qc4wyals_nested_oof5_fold${fold}_scratch_cos2_aux80_bank512_s${seed}_j${SLURM_ARRAY_JOB_ID}
task_tmp=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/qc4_oof_${fold}.XXXXXX")
train_pack=$task_tmp/train.npz
inner_pack=$task_tmp/inner_val.npz
outer_pack=$task_tmp/outer.npz
manifest=$oof_root/fold${fold}_manifest.json

mkdir -p "$oof_root"
for path in "$source_train" "$aux" "$sherlock" "$lb2_len" "$lb2_clean" "$lb2_mixed"; do
  [[ -f "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done

"$meg_python" "$brain_root/scripts/build_grouped_meg_oof_split.py" \
  --input-npz "$source_train" \
  --output-train "$train_pack" \
  --output-inner-val "$inner_pack" \
  --output-outer "$outer_pack" \
  --manifest-json "$manifest" \
  --outer-fold "$fold" \
  --num-folds 5 \
  --group-key session \
  --inner-target-rows 110

# This is the strongest practical qc4wyals-like scratch recipe.  We explicitly
# do not initialize from qc4wyals (or its Tang-trained parent), because either
# checkpoint has already seen every outer-fold session.
cd "$meg_root"
env \
  WANDB_PROJECT=braindiffusion_brain2sem \
  WANDB_JOB_TYPE=train \
  WANDB_GROUP=qc4wyals_nested_session_oof_20260908 \
  WANDB_TAGS=tang,themoth,ada002,nested-oof,session-grouped,no-outer-selection \
  CHECKPOINT_MONITOR=val_nDCG_epoch \
  CHECKPOINT_MODE=max \
  CHECKPOINT_SAVE_TOP_K=5 \
  RUN_NAME="$run_name" \
  RUN_BOOKS=1 \
  SEED="$seed" \
  PACKED_NPZ="$train_pack" \
  AUX_PACKED_NPZ="$aux" \
  VAL_PACKED_NPZ="$inner_pack" \
  TEST_PACKED_NPZ="$inner_pack" \
  SEGMENT_MS=10000 \
  EMBEDDING_TYPE=ADA \
  EMBEDDING_DIM=1536 \
  BRAIN_MODEL_ARCH=brainmagick \
  BRAIN_CONV_CHANNELS=320 \
  BRAIN_CONV_LAYERS=10 \
  BRAIN_CONV_DILATION_GROWTH=2 \
  AGGREGATION=avg \
  MODEL_NORM=batch \
  USE_PROJECTION_HEAD=1 \
  UNIQUE_WORD_BATCHES=1 \
  USE_NEGATIVE_QUEUE=0 \
  NEGATIVE_QUEUE_SIZE=8192 \
  BATCH_SIZE=60 \
  ACCUMULATE_GRAD_BATCHES=1 \
  MAX_EPOCHS="${OOF_MAX_EPOCHS:-120}" \
  PRECISION=bf16-mixed \
  LEARNING_RATE=0.0001 \
  DROPOUT=0.3 \
  AUX_FRACTION=0.80 \
  AUX_LOSS_MODE=all \
  LOSS_TYPE=clip \
  CONTRASTIVE_WEIGHT=1.0 \
  COSINE_WEIGHT=2.0 \
  VICREG_WEIGHT=0.0 \
  NORM_LOSS_WEIGHT=0.0 \
  SEMANTIC_NEGATIVE_BANK_NPZ="$banks" \
  SEMANTIC_NEGATIVE_BANK_MODE=hard_mixed \
  SEMANTIC_NEGATIVE_BANK_SIZE=512 \
  SEMANTIC_NEGATIVE_BANK_TOPK=16 \
  SEMANTIC_NEGATIVE_BANK_START_EPOCH=0 \
  SEMANTIC_NEGATIVE_BANK_EXCLUDE_POSITIVE_THRESHOLD=0.9995 \
  SEMANTIC_NEGATIVE_BANK_MAX_ROWS=0 \
  NOTES="Nested session OOF fold $fold. Outer sessions are absent from train, inner selection, initialization, and checkpoint selection. Scratch qc4wyals-like cos2/aux80 recipe." \
  bash "$meg_root/task-submit-sherlock-sva-ada002.sh"

checkpoint=$(
  "$meg_python" - "$meg_root" "$run_name" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_name = sys.argv[2]
pattern = re.compile(r"val_nDCG_epoch=([-+0-9.eE]+)\.ckpt$")
rows = []
for path in root.glob(f"wandb/*/checkpoints/{run_name}-epoch=*-val_nDCG_epoch=*.ckpt"):
    match = pattern.search(path.name)
    if match:
        rows.append((float(match.group(1)), path.stat().st_mtime_ns, path))
if not rows:
    raise SystemExit(f"No checkpoints found for {run_name}")
print(max(rows, key=lambda row: (row[0], row[1]))[2].resolve())
PY
)
printf '%s\n' "$checkpoint" > "$oof_root/fold${fold}_selected_checkpoint.txt"

"$meg_python" "$brain_root/scripts/export_qc4wyals_split_predictions.py" \
  --meg2sem-repo "$meg_root" \
  --checkpoint "$checkpoint" \
  --single-npz "$outer_pack" \
  --single-name "oof_outer_fold${fold}" \
  --output-dir "$oof_root" \
  --batch-size 8 \
  --device cuda

echo "completed fold=$fold checkpoint=$checkpoint output=$oof_root/qc4wyals_oof_outer_fold${fold}_predicted_ada002.npz"
