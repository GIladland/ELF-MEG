#!/usr/bin/env bash
#SBATCH --job-name=elfm_oraword
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=submit-jobs/train_qc4wyals_elfm_oracle_words_20260911-%j.out
#SBATCH --error=submit-jobs/train_qc4wyals_elfm_oracle_words_20260911-%j.err

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
run_tag=meg_elfm_lexwinner_oracleword_random124_frozenbase_ep30_seed49_20260911
output_dir=$runs_root/$run_tag

for required in "$python_bin" "$train_npz" "$val_npz" "$meg2sem" "$elf_oracle" "$robust_projector" "$lexical_winner"; do
  [[ -f "$required" ]] || { echo "Missing $required" >&2; exit 2; }
done
mkdir -p "$output_dir" "$data_root/tmp" "$data_root/wandb" "$data_root/wandb-data"
export WANDB_ENTITY=giladland-university-of-oxford WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=meg_elfm_oracle_known_words_20260911 WANDB_RUN_NAME=$run_tag
export WANDB_DIR=$data_root/wandb HF_HOME=$data_root/huggingface TOKENIZERS_PARALLELISM=false
export WANDB_DATA_DIR=$data_root/wandb-data
export PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=$data_root/tmp

cd "$project_root"
exec "$python_bin" scripts/train_npz_meg_to_frozen_elf.py \
  --train-npz "$train_npz" --val-npz "$val_npz" --output_dir "$output_dir" \
  --checkpoint_path "$elf_oracle" --elf-checkpoint-path "$elf_oracle" \
  --semantic-projector-checkpoint "$robust_projector" --init-e2e-checkpoint "$lexical_winner" \
  --semantic-projector-kind flat --model ELF-M --adapter-kind meg2sem_projector \
  --meg2sem-checkpoint "$meg2sem" --meg2sem-output-normalization always \
  --meg-standardization none --meg-clip-boundary -1 --freeze-meg2sem-batchnorm-stats \
  --oracle-known-word-conditioning --oracle-word-train-layout random \
  --oracle-word-eval-layout center2 --oracle-word-max-positions 10 \
  --oracle-word-expected-count 10 --oracle-word-attention-heads 8 \
  --oracle-word-dropout 0.0 --oracle-word-freeze-base --oracle-word-lr 1e-4 \
  --checkpoint-metric word_content_mean --device cuda --num-examples 0 \
  --eval-num-examples 110 --retrieval-num-examples 110 --batch_size 8 --epochs 30 \
  --eval_every 1000 --retrieval-eval-every 0 --retrieval-batch-size 64 \
  --target-encode-batch-size 64 --num_sampling_steps 32 \
  --train-timestep-mode sampling_schedule --train-timestep-steps 32 \
  --context_length 64 --semantic-hidden-dim 4096 --lr 1e-4 --weight_decay 1e-2 \
  --denoiser_loss_weight 1.0 --decoder_loss_weight 1.0 \
  --generation-t5-retrieval --interface-monitor-every 0 --skip-final-checkpoint \
  --seed 49 --use_wandb --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY" --wandb_group "$WANDB_RUN_GROUP" \
  --wandb_run_name "$WANDB_RUN_NAME" \
  --wandb_notes "LEAKY VAL110 ORACLE CEILING: exact target-derived word identities enter flow conditioning. Frozen lexical winner/MEG2SEM/projector; train only ordered T5 word-memory fusion on train2652. Protected test26 absent."
