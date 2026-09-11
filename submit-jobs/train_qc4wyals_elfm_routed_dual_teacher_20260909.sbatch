#!/usr/bin/env bash
#SBATCH --job-name=meg_dualroute
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --array=0-3%4
#SBATCH --output=submit-jobs/train_qc4wyals_elfm_routed_dual_teacher_20260909-%A_%a.out
#SBATCH --error=submit-jobs/train_qc4wyals_elfm_routed_dual_teacher_20260909-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
runs_root=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
packed=$data_root/tang_themoth_apples_exact_ada002_textmid_seg10_v1/packed
train_npz=$packed/tang_themoth_exact384_ada002_textmid_seg10000_train_dedup_fp16.npz
val_npz=$packed/tang_themoth_exact384_ada002_textmid_seg10000_val_fp16.npz
meg2sem=/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/qc4wyals/checkpoints/tang_cosine0831_ada_clip_drop03init_cos2_aux80_bank512_s82-epoch=48-val_nDCG_epoch=0.4828.ckpt
oracle=$runs_root/elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906/best.pt
robust=$runs_root/meg_elfm_ctxdistill_robustinit_exactteacher_outputprojector_ep100_seed49_20260906/best_adapter.pt
rank_teacher=$runs_root/elf_archscale_meg_ada_x16_elfm_continue_const1e5_ep8_wcmean_seed7_20260906_resid_adapter_val110_wcmean_seed49_truee2e_allmeg_projector_lora4last4_val110_wcmean_seed49/best_adapter.pt

detach_context=1
lora_rank=0
content_token=2.0
content_bow=0.10
content_precision=0.05
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) arm=split_rankctx_frozenelf ;;
  1) arm=split_rankctx_lora4; lora_rank=4 ;;
  2) arm=split_rankctx_lora4_content; lora_rank=4; content_token=3.0; content_bow=0.20; content_precision=0.10 ;;
  3) arm=joint_rankctx_lora4_control; lora_rank=4; detach_context=0 ;;
  *) exit 2 ;;
esac

for required in "$python_bin" "$train_npz" "$val_npz" "$meg2sem" "$oracle" "$robust" "$rank_teacher"; do
  [[ -f "$required" ]] || { echo "Missing $required" >&2; exit 2; }
done
run_tag=meg_elfm_dualteacher_${arm}_ep120_seed49_20260909
output_dir=$runs_root/$run_tag
mkdir -p "$output_dir" "$data_root/tmp" "$data_root/wandb" "$data_root/wandb-data"
export WANDB_ENTITY=giladland-university-of-oxford WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=meg_elfm_routed_dual_teacher_20260909 WANDB_RUN_NAME=$run_tag
export WANDB_DIR=$data_root/wandb WANDB_DATA_DIR=$data_root/wandb-data
export HF_HOME=$data_root/huggingface TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=$data_root/tmp

cd "$project_root"
args=(
  scripts/train_npz_meg_to_frozen_elf.py
  --train-npz "$train_npz" --val-npz "$val_npz" --output_dir "$output_dir"
  --checkpoint_path "$oracle" --elf-checkpoint-path "$oracle"
  --semantic-projector-checkpoint "$oracle" --semantic-context-teacher-checkpoint "$oracle"
  --init-e2e-checkpoint "$robust" --semantic-rank-teacher-e2e-checkpoint "$rank_teacher"
  --model ELF-M --adapter-kind meg2sem_projector --meg2sem-checkpoint "$meg2sem"
  --meg2sem-output-normalization always --meg-standardization none --meg-clip-boundary -1
  --train-meg2sem --meg2sem-unfreeze-scope output --train-semantic-projector
  --freeze-meg2sem-batchnorm-stats --device cuda --num-examples 0
  --eval-num-examples 110 --retrieval-num-examples 110 --batch_size 8 --epochs 120
  --eval_every 500 --retrieval-eval-every 500 --retrieval-batch-size 64
  --target-encode-batch-size 64 --num_sampling_steps 32
  --train-timestep-mode sampling_schedule --train-timestep-steps 32
  --context_length 64 --semantic-hidden-dim 4096
  --semantic-alignment-loss-weight 1.0 --semantic-alignment-loss-type cosine
  --semantic-contrastive-loss-weight 0.20 --semantic-contrastive-temperature 0.07
  --semantic-context-loss-weight 2.0 --semantic-rank-distill-loss-weight 0.25
  --semantic-rank-distill-temperature 0.07
  --decoder-content-token-weight "$content_token"
  --decoder-content-bow-loss-weight "$content_bow"
  --decoder-content-precision-loss-weight "$content_precision"
  --decoder-condition-pairing-margin-weight 0.05 --decoder-condition-pairing-margin 0.10
  --lr 1e-6 --semantic-projector-lr 1e-6 --meg2sem-lr 2e-8
  --weight_decay 1e-2 --cfg_scale 1.0 --self_cond_cfg_scale 1.0
  --cond_dropout_prob 0.0 --denoiser_loss_weight 1.0 --decoder_loss_weight 1.0
  --train-target-mask-mode full --interface-monitor-every 50 --interface-monitor-batch-size 32
  --generation-t5-retrieval --oracle-semantic-eval --checkpoint-metric word_content_mean
  --skip-final-checkpoint --seed 49 --use_wandb --wandb_project "$WANDB_PROJECT"
  --wandb_entity "$WANDB_ENTITY" --wandb_group "$WANDB_RUN_GROUP"
  --wandb_run_name "$WANDB_RUN_NAME"
  --wandb_notes "Dual teacher: exact ADA teaches ELF context; frozen candidate-3 MEG adapter teaches retrieval logits. Split arms stop lexical gradients at MEG output. Train2652/val110 only; test26 absent."
)
if [[ "$detach_context" == 1 ]]; then
  args+=(--detach-meg2sem-for-context)
fi
if [[ "$lora_rank" -gt 0 ]]; then
  args+=(--elf-lora-rank "$lora_rank" --elf-lora-alpha 8 --elf-lora-dropout 0.05 \
    --elf-lora-targets attention,mlp --elf-lora-last-n-blocks 4 --elf-lr 2e-7)
fi
"$python_bin" "${args[@]}"

bert_root=$data_root/python-packages/bert-score-0.3.13
export PYTHONPATH=$bert_root${PYTHONPATH:+:$PYTHONPATH}
"$python_bin" scripts/add_bertscore_to_metrics.py \
  --metrics-json "$output_dir/best_metrics.json" \
  --output "$output_dir/best_metrics.bertscore.json" --batch-size 8 --device cuda --top-k 10
