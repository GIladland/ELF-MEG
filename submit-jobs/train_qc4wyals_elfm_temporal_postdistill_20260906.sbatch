#!/usr/bin/env bash
#SBATCH --job-name=elfm_tmppost
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-2%3
#SBATCH --output=submit-jobs/train_qc4wyals_elfm_temporal_postdistill_20260906-%A_%a.out
#SBATCH --error=submit-jobs/train_qc4wyals_elfm_temporal_postdistill_20260906-%A_%a.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
packed=$data_root/tang_themoth_apples_exact_ada002_textmid_seg10_v1/packed
train_npz=$packed/tang_themoth_exact384_ada002_textmid_seg10000_train_dedup_fp16.npz
val_npz=$packed/tang_themoth_exact384_ada002_textmid_seg10000_val_fp16.npz
oracle=/data/engs-pnpl/glandau/elf-runs/elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906/best.pt
initial=/data/engs-pnpl/glandau/elf-runs/meg_elfm_temporal64_ctxdistill_exactteacher_ep100_seed49_20260906/best_adapter.pt
lora=0
lora_last=4
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) arm=frozenelf ;;
  1) arm=lora4last4; lora=4 ;;
  2) arm=lora8last8; lora=8; lora_last=8 ;;
  *) exit 2 ;;
esac
run_tag=meg_elfm_temporal64_postctx_${arm}_strongcontent_ep80_seed49_20260906
output_dir=/data/engs-pnpl/glandau/elf-runs/$run_tag
for required in "$python_bin" "$train_npz" "$val_npz" "$oracle" "$initial"; do
  [[ -f "$required" ]] || { echo "Missing $required" >&2; exit 2; }
done
mkdir -p "$output_dir" "$data_root/tmp" "$data_root/wandb" "$data_root/wandb-data"
export WANDB_ENTITY=giladland-university-of-oxford WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=meg_elfm_temporal_post_context_20260906 WANDB_RUN_NAME=$run_tag
export WANDB_DIR=$data_root/wandb HF_HOME=$data_root/huggingface TOKENIZERS_PARALLELISM=false
export WANDB_DATA_DIR=$data_root/wandb-data
export PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=$data_root/tmp
cd "$project_root"
args=(
  scripts/train_npz_meg_to_frozen_elf.py
  --train-npz "$train_npz" --val-npz "$val_npz" --output_dir "$output_dir"
  --checkpoint_path "$oracle" --elf-checkpoint-path "$oracle"
  --semantic-projector-checkpoint "$oracle" --model ELF-M --adapter-kind meg_context
  --init-adapter-checkpoint "$initial" --meg-standardization none --meg-clip-boundary -1
  --device cuda --num-examples 0 --eval-num-examples 110 --retrieval-num-examples 110
  --batch_size 8 --epochs 80 --eval_every 500 --retrieval-eval-every 500
  --retrieval-batch-size 64 --target-encode-batch-size 64 --num_sampling_steps 32
  --train-timestep-mode sampling_schedule --train-timestep-steps 32
  --context_length 64 --semantic-hidden-dim 4096 --lr 2e-6 --weight_decay 1e-2
  --adapter-dropout 0.05 --attention-heads 8 --decoder-content-token-weight 4
  --decoder-content-bow-loss-weight 0.25 --decoder-content-precision-loss-weight 0.10
  --decoder-condition-pairing-margin-weight 0.10 --decoder-condition-pairing-margin 0.10
  --cfg_scale 1.0 --self_cond_cfg_scale 1.0 --cond_dropout_prob 0.0
  --denoiser_loss_weight 1.0 --decoder_loss_weight 1.0 --train-target-mask-mode full
  --generation-t5-retrieval --oracle-semantic-eval --checkpoint-metric word_content_mean
  --skip-final-checkpoint --seed 49 --use_wandb --wandb_project "$WANDB_PROJECT"
  --wandb_entity "$WANDB_ENTITY" --wandb_group "$WANDB_RUN_GROUP"
  --wandb_run_name "$WANDB_RUN_NAME"
  --wandb_notes "Stage B direct temporal context model from frozen teacher distillation; strong lexical losses; train2652/val110, test26 absent."
)
if [[ "$lora" -gt 0 ]]; then
  args+=(--elf-lora-rank "$lora" --elf-lora-alpha "$((lora * 2))" --elf-lora-dropout 0.05 \
    --elf-lora-targets attention,mlp --elf-lora-last-n-blocks "$lora_last" --elf-lr 5e-7)
fi
exec "$python_bin" "${args[@]}"
