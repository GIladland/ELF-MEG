#!/usr/bin/env bash
#SBATCH --job-name=elf_meg_ml_x16e2e
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-7%4
#SBATCH --output=submit-jobs/train_elf_archscale_meg_ml_x16_true_e2e_search_20260906-%A_%a.out
#SBATCH --error=submit-jobs/train_elf_archscale_meg_ml_x16_true_e2e_search_20260906-%A_%a.err

set -euo pipefail

# True raw-MEG E2E on the promoted x16-backed M/L adapter.  Compare updating
# only the MEG2SEM semantic readout versus all MEG2SEM parameters, each with
# frozen ELF or final-four-block rank-4 LoRA.  val110 is selection-only and
# protected test26 brain vectors are absent.
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
packed_root=$data_root/tang_themoth_apples_exact_ada002_textmid_seg10_v1/packed
train_npz=$packed_root/tang_themoth_exact384_ada002_textmid_seg10000_train_dedup_fp16.npz
val_npz=$packed_root/tang_themoth_exact384_ada002_textmid_seg10000_val_fp16.npz
meg2sem_checkpoint=/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/qc4wyals/checkpoints/tang_cosine0831_ada_clip_drop03init_cos2_aux80_bank512_s82-epoch=48-val_nDCG_epoch=0.4828.ckpt

task=${SLURM_ARRAY_TASK_ID:-0}
model_index=$((task / 4))
variant_index=$((task % 4))
case "$model_index" in
  0) model=ELF-M; model_slug=elfm; recipes=(continue_const1e5 continue_cos2e5) ;;
  1) model=ELF-L; model_slug=elfl; recipes=(continue_const5e6 continue_cos1e5) ;;
  *) exit 2 ;;
esac

selector_args=(scripts/select_best_elf_candidate.py --metric word_content_mean --format tsv)
for recipe in "${recipes[@]}"; do
  for adapter_label in flat resid; do
    tag=elf_archscale_meg_ada_x16_${model_slug}_${recipe}_ep8_wcmean_seed7_20260906_${adapter_label}_adapter_val110_wcmean_seed49
    selector_args+=(--candidate "$tag,$model,/data/engs-pnpl/glandau/elf-runs/$tag")
  done
done

cd "$project_root"
IFS=$'\t' read -r selected_tag selected_model selected_dir selected_checkpoint selected_score < <(
  "$python_bin" "${selector_args[@]}"
)
if [[ "$selected_model" != "$model" || ! -f "$selected_checkpoint" ]]; then
  echo "Invalid promoted candidate: model=$selected_model checkpoint=$selected_checkpoint" >&2
  exit 3
fi
case "$selected_tag" in
  *_flat_adapter_*) projector_kind=flat; mapper_hidden=1024 ;;
  *_resid_adapter_*) projector_kind=residual_identity; mapper_hidden=1024 ;;
  *) echo "Cannot infer MEG projector kind from $selected_tag" >&2; exit 4 ;;
esac

use_lora=0
case "$variant_index" in
  0) arm=output_projector_frozenelf; meg_scope=output; meg_lr=5e-8 ;;
  1) arm=output_projector_lora4last4; meg_scope=output; meg_lr=5e-8; use_lora=1 ;;
  2) arm=allmeg_projector_frozenelf; meg_scope=all; meg_lr=5e-9 ;;
  3) arm=allmeg_projector_lora4last4; meg_scope=all; meg_lr=5e-9; use_lora=1 ;;
  *) exit 2 ;;
esac

for required in "$python_bin" "$train_npz" "$val_npz" "$meg2sem_checkpoint" "$selected_checkpoint"; do
  if [[ ! -e "$required" ]]; then echo "Missing required path: $required" >&2; exit 5; fi
done

echo "Promoted $selected_tag score=$selected_score checkpoint=$selected_checkpoint"
run_tag=${selected_tag}_truee2e_${arm}_val110_wcmean_seed49
output_dir=/data/engs-pnpl/glandau/elf-runs/$run_tag
mkdir -p "$output_dir" "$data_root/tmp" "$data_root/wandb" "$data_root/wandb-data" "$data_root/wandb-cache"

export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=elf_archscale_meg_ml_x16_true_e2e_20260906
export WANDB_RUN_NAME=$run_tag
export WANDB_DIR=$data_root/wandb
export WANDB_DATA_DIR=$data_root/wandb-data
export WANDB_CACHE_DIR=$data_root/wandb-cache
export XDG_CACHE_HOME=$data_root/xdg-cache
export HF_HOME=$data_root/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TORCH_HOME=$data_root/torch
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=$data_root/tmp

args=(
  scripts/train_npz_meg_to_frozen_elf.py
  --train-npz "$train_npz"
  --val-npz "$val_npz"
  --output_dir "$output_dir"
  --checkpoint_path "$selected_checkpoint"
  --elf-checkpoint-path "$selected_checkpoint"
  --semantic-projector-checkpoint "$selected_checkpoint"
  --semantic-projector-kind "$projector_kind"
  --semantic-projector-mapper-hidden-dim "$mapper_hidden"
  --semantic-projector-dropout 0.0
  --model "$model"
  --adapter-kind meg2sem_projector
  --meg2sem-checkpoint "$meg2sem_checkpoint"
  --meg2sem-output-normalization always
  --meg-standardization none
  --meg-clip-boundary -1
  --train-meg2sem
  --meg2sem-unfreeze-scope "$meg_scope"
  --freeze-meg2sem-batchnorm-stats
  --train-semantic-projector
  --device cuda
  --num-examples 0
  --eval-num-examples 110
  --retrieval-num-examples 110
  --batch_size 8
  --epochs 60
  --eval_every 500
  --retrieval-eval-every 500
  --retrieval-batch-size 64
  --target-encode-batch-size 64
  --num_sampling_steps 32
  --train-timestep-mode sampling_schedule
  --train-timestep-steps 32
  --context_length 64
  --semantic-hidden-dim 4096
  --semantic-alignment-loss-weight 2.0
  --semantic-alignment-loss-type cosine
  --lr 1e-6
  --semantic-projector-lr 1e-6
  --meg2sem-lr "$meg_lr"
  --weight_decay 1e-2
  --cfg_scale 1.0
  --self_cond_cfg_scale 1.0
  --cond_dropout_prob 0.0
  --denoiser_loss_weight 1.0
  --decoder_loss_weight 1.0
  --decoder-content-token-weight 2.0
  --decoder-content-bow-loss-weight 0.10
  --decoder-content-precision-loss-weight 0.05
  --decoder-condition-pairing-margin-weight 0.05
  --decoder-condition-pairing-margin 0.1
  --train-target-mask-mode full
  --interface-monitor-every 50
  --interface-monitor-batch-size 32
  --generation-t5-retrieval
  --oracle-semantic-eval
  --checkpoint-metric word_content_mean
  --skip-final-checkpoint
  --seed 49
  --use_wandb
  --wandb_project "$WANDB_PROJECT"
  --wandb_entity "$WANDB_ENTITY"
  --wandb_group "$WANDB_RUN_GROUP"
  --wandb_run_name "$WANDB_RUN_NAME"
  --wandb_sample_examples 16
  --wandb_notes "True raw-MEG E2E from qc4wyals plus promoted x16 $model adapter. Arm=$arm. Train2652/val110 only; protected test26 absent. Selection=mean word/content F1."
)

if [[ "$use_lora" == 1 ]]; then
  args+=(
    --elf-lora-rank 4
    --elf-lora-alpha 8
    --elf-lora-dropout 0.05
    --elf-lora-targets attention,mlp
    --elf-lora-last-n-blocks 4
    --elf-lr 2e-7
  )
fi

"$python_bin" "${args[@]}"

bert_root=$data_root/python-packages/bert-score-0.3.13
export PYTHONPATH=$bert_root${PYTHONPATH:+:$PYTHONPATH}
"$python_bin" scripts/add_bertscore_to_metrics.py \
  --metrics-json "$output_dir/best_metrics.json" \
  --output "$output_dir/best_metrics.bertscore.json" \
  --batch-size 8 --device cuda --top-k 10
