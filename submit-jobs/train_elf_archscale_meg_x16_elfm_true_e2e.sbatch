#!/usr/bin/env bash
#SBATCH --job-name=elfm_cos_e2e
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-1%2
#SBATCH --output=submit-jobs/train_elf_archscale_meg_x16_elfm_true_e2e-%A_%a.out
#SBATCH --error=submit-jobs/train_elf_archscale_meg_x16_elfm_true_e2e-%A_%a.err

set -euo pipefail

# True raw-MEG E2E comparison on the new cosine-schedule x16 ELF-M winner:
#   task 0: MEG2SEM output head + semantic projector, frozen ELF
#   task 1: the same path plus rank-4 LoRA in ELF's final four blocks
# Both fit train2652, select on val110, and never load protected test26 MEG.
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
packed_root=$data_root/tang_themoth_apples_exact_ada002_textmid_seg10_v1/packed
train_npz=$packed_root/tang_themoth_exact384_ada002_textmid_seg10000_train_dedup_fp16.npz
val_npz=$packed_root/tang_themoth_exact384_ada002_textmid_seg10000_val_fp16.npz
meg2sem_checkpoint=/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/qc4wyals/checkpoints/tang_cosine0831_ada_clip_drop03init_cos2_aux80_bank512_s82-epoch=48-val_nDCG_epoch=0.4828.ckpt
oracle_checkpoint=/data/engs-pnpl/glandau/elf-runs/elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906/best.pt
continuation_checkpoint=${CONTINUATION_CHECKPOINT:-}

for required in "$python_bin" "$train_npz" "$val_npz" "$meg2sem_checkpoint" "$oracle_checkpoint"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 2
  fi
done
if [[ -n "$continuation_checkpoint" && ! -f "$continuation_checkpoint" ]]; then
  echo "Missing continuation checkpoint: $continuation_checkpoint" >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) arm=outputmeg_projector_frozenelf; use_lora=0 ;;
  1) arm=outputmeg_projector_lora4last4; use_lora=1 ;;
  *) exit 2 ;;
esac

run_tag=${RUN_TAG_OVERRIDE:-elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906_truee2e_${arm}_val110_seed49}
output_dir=/data/engs-pnpl/glandau/elf-runs/$run_tag
mkdir -p "$output_dir" "$data_root/tmp" "$data_root/wandb" "$data_root/wandb-data" "$data_root/wandb-cache"

export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP_OVERRIDE:-elf_archscale_meg_x16_coswinner_e2e_20260906}
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

cd "$project_root"
args=(
  scripts/train_npz_meg_to_frozen_elf.py
  --train-npz "$train_npz"
  --val-npz "$val_npz"
  --output_dir "$output_dir"
  --checkpoint_path "$oracle_checkpoint"
  --elf-checkpoint-path "$oracle_checkpoint"
  --semantic-projector-checkpoint "$oracle_checkpoint"
  --model ELF-M
  --adapter-kind meg2sem_projector
  --meg2sem-checkpoint "$meg2sem_checkpoint"
  --meg2sem-output-normalization always
  --meg-standardization none
  --meg-clip-boundary -1
  --train-meg2sem
  --meg2sem-unfreeze-scope output
  --freeze-meg2sem-batchnorm-stats
  --train-semantic-projector
  --device cuda
  --num-examples 0
  --eval-num-examples 110
  --retrieval-num-examples 110
  --batch_size 8
  --epochs "${EPOCHS_OVERRIDE:-20}"
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
  --lr 5e-6
  --semantic-projector-lr 5e-6
  --meg2sem-lr 5e-8
  --weight_decay 1e-2
  --cfg_scale 1.0
  --self_cond_cfg_scale 1.0
  --cond_dropout_prob 0.0
  --denoiser_loss_weight 1.0
  --decoder_loss_weight 1.0
  --train-target-mask-mode full
  --interface-monitor-every 50
  --interface-monitor-batch-size 32
  --generation-t5-retrieval
  --oracle-semantic-eval
  --checkpoint-metric word_content_mean
  --seed 49
  --skip-final-checkpoint
  --use_wandb
  --wandb_project "$WANDB_PROJECT"
  --wandb_entity "$WANDB_ENTITY"
  --wandb_group "$WANDB_RUN_GROUP"
  --wandb_run_name "$WANDB_RUN_NAME"
  --wandb_sample_examples 16
  --wandb_notes "True raw-MEG E2E initialized from the x16 cosine ELF-M winner or CONTINUATION_CHECKPOINT when supplied (exact ADA locked eval word=.9109 content=.9054). Update only MEG2SEM semantic output, semantic projector, and optional final-four-block ELF LoRA. Train2652/val110; protected test26 absent."
)

if [[ -n "$continuation_checkpoint" ]]; then
  args+=(--init-e2e-checkpoint "$continuation_checkpoint")
fi

if [[ "$use_lora" == 1 ]]; then
  args+=(
    --elf-lora-rank 4
    --elf-lora-alpha 8
    --elf-lora-dropout 0.05
    --elf-lora-targets attention,mlp
    --elf-lora-last-n-blocks 4
    --elf-lr 1e-6
  )
fi

"$python_bin" "${args[@]}"

bert_root=$data_root/python-packages/bert-score-0.3.13
export PYTHONPATH=$bert_root${PYTHONPATH:+:$PYTHONPATH}
"$python_bin" scripts/add_bertscore_to_metrics.py \
  --metrics-json "$output_dir/best_metrics.json" \
  --output "$output_dir/best_metrics.bertscore.json" \
  --batch-size 8 --device cuda --top-k 10
