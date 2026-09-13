#!/usr/bin/env bash
#SBATCH --job-name=elfm_orasmoke
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=submit-jobs/smoke_qc4wyals_elfm_oracle_words_20260911-%j.out
#SBATCH --error=submit-jobs/smoke_qc4wyals_elfm_oracle_words_20260911-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
runs_root=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
packed=$data_root/tang_themoth_apples_exact_ada002_textmid_seg10_v1/packed
train_npz=$packed/tang_themoth_exact384_ada002_textmid_seg10000_train_dedup_fp16.npz
val_npz=$packed/tang_themoth_exact384_ada002_textmid_seg10000_val_fp16.npz
meg2sem=/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/qc4wyals/checkpoints/tang_cosine0831_ada_clip_drop03init_cos2_aux80_bank512_s82-epoch=48-val_nDCG_epoch=0.4828.ckpt
elf_oracle=$runs_root/elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906/best.pt
robust_projector=$runs_root/elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906_flat_adapter_val110_wcmean_seed49/best.pt
lexical_winner=$runs_root/meg_elfm_ctxdistill_robustinit_exactteacher_outputprojector_ep100_seed49_20260906/best_adapter.pt
output_dir=$runs_root/meg_elfm_oracleword_smoke_${SLURM_JOB_ID}

mkdir -p "$output_dir" "$data_root/tmp"
export HF_HOME=$data_root/huggingface TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=$data_root/tmp

cd "$project_root"
exec "$python_bin" scripts/train_npz_meg_to_frozen_elf.py \
  --train-npz "$train_npz" --val-npz "$val_npz" --output_dir "$output_dir" \
  --checkpoint_path "$elf_oracle" --elf-checkpoint-path "$elf_oracle" \
  --semantic-projector-checkpoint "$robust_projector" --init-e2e-checkpoint "$lexical_winner" \
  --semantic-projector-kind flat --model ELF-M --adapter-kind meg2sem_projector \
  --meg2sem-checkpoint "$meg2sem" --meg2sem-output-normalization always \
  --meg-standardization none --meg-clip-boundary -1 --freeze-meg2sem-batchnorm-stats \
  --oracle-known-word-conditioning --oracle-word-train-layout random \
  --oracle-word-eval-layout center2 --oracle-word-freeze-base --oracle-word-lr 1e-4 \
  --checkpoint-metric word_content_mean --device cuda --num-examples 8 \
  --eval-num-examples 1 --retrieval-num-examples 1 --batch_size 2 --steps 1 \
  --eval_every 1 --retrieval-eval-every 0 --target-encode-batch-size 8 \
  --num_sampling_steps 2 --train-timestep-mode sampling_schedule \
  --train-timestep-steps 2 --context_length 64 --semantic-hidden-dim 4096 \
  --generation-t5-retrieval --interface-monitor-every 0 --skip-final-checkpoint --seed 49
