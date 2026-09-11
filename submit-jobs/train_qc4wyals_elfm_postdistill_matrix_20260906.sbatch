#!/usr/bin/env bash
#SBATCH --job-name=elfm_postctx
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-5%4
#SBATCH --output=submit-jobs/train_qc4wyals_elfm_postdistill_matrix_20260906-%A_%a.out
#SBATCH --error=submit-jobs/train_qc4wyals_elfm_postdistill_matrix_20260906-%A_%a.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
packed=$data_root/tang_themoth_apples_exact_ada002_textmid_seg10_v1/packed
train_npz=$packed/tang_themoth_exact384_ada002_textmid_seg10000_train_dedup_fp16.npz
val_npz=$packed/tang_themoth_exact384_ada002_textmid_seg10000_val_fp16.npz
meg2sem=/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/qc4wyals/checkpoints/tang_cosine0831_ada_clip_drop03init_cos2_aux80_bank512_s82-epoch=48-val_nDCG_epoch=0.4828.ckpt
oracle=/data/engs-pnpl/glandau/elf-runs/elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906/best.pt
distilled=/data/engs-pnpl/glandau/elf-runs/meg_elfm_ctxdistill_selected_20260906/selected_context_adapter.pt

train_meg=1
train_projector=1
meg_scope=output
meg_lr=2e-8
projector_lr=2e-6
lora_rank=0
lora_last=4
elf_lr=5e-7
content_token=2
content_bow=0.10
content_precision=0.05
pairing=0.05
semantic_context=2.0
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) arm=frozenelf_balanced ;;
  1) arm=lora4last4_balanced; lora_rank=4 ;;
  2) arm=lora4last4_strongcontent; lora_rank=4; content_token=4; content_bow=0.25; content_precision=0.10; pairing=0.10 ;;
  3) arm=frozenadapter_lora4last4_strongcontent; train_meg=0; train_projector=0; lora_rank=4; content_token=4; content_bow=0.25; content_precision=0.10; pairing=0.10; semantic_context=0 ;;
  4) arm=lora8last8_strongcontent; lora_rank=8; lora_last=8; content_token=4; content_bow=0.25; content_precision=0.10; pairing=0.10 ;;
  5) arm=fullmeg_lora4last4_balanced; meg_scope=all; meg_lr=1e-8; lora_rank=4 ;;
  *) exit 2 ;;
esac

for required in "$python_bin" "$train_npz" "$val_npz" "$meg2sem" "$oracle" "$distilled"; do
  [[ -f "$required" ]] || { echo "Missing $required" >&2; exit 2; }
done
run_tag=meg_elfm_postctx_${arm}_ep80_seed49_20260906
output_dir=/data/engs-pnpl/glandau/elf-runs/$run_tag
mkdir -p "$output_dir" "$data_root/tmp" "$data_root/wandb" "$data_root/wandb-data"
export WANDB_ENTITY=giladland-university-of-oxford WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=meg_elfm_post_context_matrix_20260906 WANDB_RUN_NAME=$run_tag
export WANDB_DIR=$data_root/wandb HF_HOME=$data_root/huggingface TOKENIZERS_PARALLELISM=false
export WANDB_DATA_DIR=$data_root/wandb-data
export PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=$data_root/tmp
cd "$project_root"
args=(
  scripts/train_npz_meg_to_frozen_elf.py
  --train-npz "$train_npz" --val-npz "$val_npz" --output_dir "$output_dir"
  --checkpoint_path "$oracle" --elf-checkpoint-path "$oracle"
  --semantic-projector-checkpoint "$oracle" --semantic-context-teacher-checkpoint "$oracle"
  --init-e2e-checkpoint "$distilled" --model ELF-M
  --adapter-kind meg2sem_projector --meg2sem-checkpoint "$meg2sem"
  --meg2sem-output-normalization always --meg-standardization none --meg-clip-boundary -1
  --freeze-meg2sem-batchnorm-stats --meg2sem-unfreeze-scope "$meg_scope"
  --device cuda --num-examples 0 --eval-num-examples 110 --retrieval-num-examples 110
  --batch_size 8 --epochs 80 --eval_every 500 --retrieval-eval-every 500
  --retrieval-batch-size 64 --target-encode-batch-size 64 --num_sampling_steps 32
  --train-timestep-mode sampling_schedule --train-timestep-steps 32
  --context_length 64 --semantic-hidden-dim 4096
  --semantic-alignment-loss-weight 1.0 --semantic-alignment-loss-type cosine
  --semantic-contrastive-loss-weight 0.10 --semantic-contrastive-temperature 0.07
  --semantic-context-loss-weight "$semantic_context"
  --semantic-geometry-loss-weight 0.10 --semantic-rank-distill-loss-weight 0.10
  --semantic-rank-distill-temperature 0.07
  --decoder-content-token-weight "$content_token"
  --decoder-content-bow-loss-weight "$content_bow"
  --decoder-content-precision-loss-weight "$content_precision"
  --decoder-condition-pairing-margin-weight "$pairing"
  --decoder-condition-pairing-margin 0.10
  --lr 2e-6 --semantic-projector-lr "$projector_lr" --meg2sem-lr "$meg_lr"
  --weight_decay 1e-2 --cfg_scale 1.0 --self_cond_cfg_scale 1.0
  --cond_dropout_prob 0.0 --denoiser_loss_weight 1.0 --decoder_loss_weight 1.0
  --train-target-mask-mode full --interface-monitor-every 50 --interface-monitor-batch-size 32
  --generation-t5-retrieval --oracle-semantic-eval --checkpoint-metric word_content_mean
  --skip-final-checkpoint --seed 49 --use_wandb --wandb_project "$WANDB_PROJECT"
  --wandb_entity "$WANDB_ENTITY" --wandb_group "$WANDB_RUN_GROUP"
  --wandb_run_name "$WANDB_RUN_NAME"
  --wandb_notes "Stage B from validation-selected frozen context distillation. Same train2652/val110 and exact frozen context teacher; test26 absent."
)
[[ "$train_meg" == 1 ]] && args+=(--train-meg2sem)
[[ "$train_projector" == 1 ]] && args+=(--train-semantic-projector)
if [[ "$lora_rank" -gt 0 ]]; then
  args+=(--elf-lora-rank "$lora_rank" --elf-lora-alpha "$((lora_rank * 2))" \
    --elf-lora-dropout 0.05 --elf-lora-targets attention,mlp \
    --elf-lora-last-n-blocks "$lora_last" --elf-lr "$elf_lr")
fi
exec "$python_bin" "${args[@]}"
