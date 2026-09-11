#!/usr/bin/env bash
#SBATCH --job-name=qc4w_splits
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=submit-jobs/export_qc4wyals_split_predictions-%j.out
#SBATCH --error=submit-jobs/export_qc4wyals_split_predictions-%j.err

set -euo pipefail

brain_root=/data/engs-pnpl/glandau/BrainDiffusion/ELF
meg_root=/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM
data_root=/data/engs-pnpl/glandau/elf-cache/tang_themoth_apples_exact_ada002_textmid_seg10_v1
checkpoint=/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/qc4wyals/checkpoints/tang_cosine0831_ada_clip_drop03init_cos2_aux80_bank512_s82-epoch=48-val_nDCG_epoch=0.4828.ckpt
output_dir=/data/engs-pnpl/glandau/elf-cache/tang_themoth_qc4wyals_ada_v2

cd "$meg_root"
exec /data/engs-pnpl/glandau/conda/envs/meg2sem/bin/python \
  "$brain_root/scripts/export_qc4wyals_split_predictions.py" \
  --meg2sem-repo "$meg_root" \
  --checkpoint "$checkpoint" \
  --train-npz "$data_root/packed/tang_themoth_exact384_ada002_textmid_seg10000_train_dedup_fp16.npz" \
  --val-npz "$data_root/packed/tang_themoth_exact384_ada002_textmid_seg10000_val_fp16.npz" \
  --test-npz "$data_root/packed/tang_themoth_exact384_ada002_textmid_seg10000_test_fp16.npz" \
  --output-dir "$output_dir" \
  --batch-size 8 \
  --device cuda
