#!/usr/bin/env bash
#SBATCH --job-name=elfm_ctxdist
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-3%4
#SBATCH --output=submit-jobs/train_qc4wyals_elfm_context_distill_20260906-%A_%a.out
#SBATCH --error=submit-jobs/train_qc4wyals_elfm_context_distill_20260906-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
packed=$data_root/tang_themoth_apples_exact_ada002_textmid_seg10_v1/packed
train_npz=$packed/tang_themoth_exact384_ada002_textmid_seg10000_train_dedup_fp16.npz
val_npz=$packed/tang_themoth_exact384_ada002_textmid_seg10000_val_fp16.npz
meg2sem=/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/qc4wyals/checkpoints/tang_cosine0831_ada_clip_drop03init_cos2_aux80_bank512_s82-epoch=48-val_nDCG_epoch=0.4828.ckpt
oracle=/data/engs-pnpl/glandau/elf-runs/elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906/best.pt

train_meg=0
train_projector=0
meg_scope=output
meg_lr=5e-8
projector_lr=5e-6
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) arm=output_only; train_meg=1 ;;
  1) arm=projector_only; train_projector=1 ;;
  2) arm=output_plus_projector; train_meg=1; train_projector=1 ;;
  3) arm=fullmeg_plus_projector; train_meg=1; train_projector=1; meg_scope=all; meg_lr=2e-8 ;;
  *) exit 2 ;;
esac

run_tag=meg_elfm_ctxdistill_exactteacher_${arm}_ep100_seed49_20260906
output_dir=/data/engs-pnpl/glandau/elf-runs/$run_tag
mkdir -p "$output_dir" "$data_root/tmp" "$data_root/wandb" "$data_root/wandb-data"

export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=meg_elfm_context_distillation_20260906
export WANDB_RUN_NAME=$run_tag
export WANDB_DIR=$data_root/wandb
export WANDB_DATA_DIR=$data_root/wandb-data
export HF_HOME=$data_root/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=$data_root/tmp

cd "$project_root"
args=(
  scripts/train_npz_meg_to_frozen_elf.py
  --train-npz "$train_npz" --val-npz "$val_npz" --output_dir "$output_dir"
  --checkpoint_path "$oracle" --elf-checkpoint-path "$oracle"
  --semantic-projector-checkpoint "$oracle" --model ELF-M
  --adapter-kind meg2sem_projector --meg2sem-checkpoint "$meg2sem"
  --meg2sem-output-normalization always --meg-standardization none --meg-clip-boundary -1
  --freeze-meg2sem-batchnorm-stats --meg2sem-unfreeze-scope "$meg_scope"
  --teacher-context-loss-weight 1.0 --checkpoint-metric teacher_context_cosine
  --device cuda --num-examples 0 --eval-num-examples 110 --retrieval-num-examples 110
  --batch_size 8 --epochs 100 --eval_every 1000 --retrieval-eval-every 1000
  --retrieval-batch-size 64 --target-encode-batch-size 64
  --num_sampling_steps 32 --train-timestep-mode sampling_schedule --train-timestep-steps 32
  --context_length 64 --semantic-hidden-dim 4096
  --lr 5e-6 --semantic-projector-lr "$projector_lr" --meg2sem-lr "$meg_lr"
  --weight_decay 1e-2 --cfg_scale 1.0 --self_cond_cfg_scale 1.0
  --generation-t5-retrieval --oracle-semantic-eval --interface-monitor-every 50
  --interface-monitor-batch-size 32 --skip-final-checkpoint --seed 49
  --use_wandb --wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY"
  --wandb_group "$WANDB_RUN_GROUP" --wandb_run_name "$WANDB_RUN_NAME"
  --wandb_notes "Stage A frozen exact-context distillation. MEG train2652 only; selection is val110 teacher-context cosine; protected test26 is absent."
)
[[ "$train_meg" == 1 ]] && args+=(--train-meg2sem)
[[ "$train_projector" == 1 ]] && args+=(--train-semantic-projector)
exec "$python_bin" "${args[@]}"
