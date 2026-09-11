#!/usr/bin/env bash
#SBATCH --job-name=qc4oof5_e2e
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-5%3
#SBATCH --output=submit-jobs/train_qc4wyals_oof5_elfb_true_e2e_matrix_20260908-%A_%a.out
#SBATCH --error=submit-jobs/train_qc4wyals_oof5_elfb_true_e2e_matrix_20260908-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=${PYTHON_BIN:-$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python}
packed_root=$data_root/tang_themoth_apples_exact_ada002_textmid_seg10_v1/packed
train_npz=$packed_root/tang_themoth_exact384_ada002_textmid_seg10000_train_dedup_fp16.npz
val_npz=$packed_root/tang_themoth_exact384_ada002_textmid_seg10000_val_fp16.npz
meg2sem_checkpoint=${MEG2SEM_CHECKPOINT:-/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/qc4wyals/checkpoints/tang_cosine0831_ada_clip_drop03init_cos2_aux80_bank512_s82-epoch=48-val_nDCG_epoch=0.4828.ckpt}
oof_flat_checkpoint=${OOF_FLAT_CHECKPOINT:-/data/engs-pnpl/glandau/elf-runs/qc4wyals_nested_oof5_elfb_flat_context_ep300_seed49_20260908/best.pt}

for required in "$python_bin" "$train_npz" "$val_npz" "$meg2sem_checkpoint" "$oof_flat_checkpoint"; do
  [[ -f "$required" ]] || { echo "Missing required path: $required" >&2; exit 2; }
done

train_meg2sem=0
meg_scope=output
meg_lr=5e-8
projector_lr=1e-6
elf_lora_rank=0
elf_lora_last_n=4
elf_lr=2e-7
semantic_alignment=0.0
content_token=2.0
content_bow=0.10
content_precision=0.05
pairing=0.05

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    arm=projector_only
    ;;
  1)
    arm=projector_lora4last4
    elf_lora_rank=4
    ;;
  2)
    arm=outputmeg_projector
    train_meg2sem=1
    semantic_alignment=2.0
    ;;
  3)
    arm=outputmeg_projector_lora4last4
    train_meg2sem=1
    semantic_alignment=2.0
    elf_lora_rank=4
    ;;
  4)
    arm=fullmeg_projector
    train_meg2sem=1
    meg_scope=all
    meg_lr=5e-9
    semantic_alignment=2.0
    ;;
  5)
    arm=fullmeg_projector_lora4last4
    train_meg2sem=1
    meg_scope=all
    meg_lr=5e-9
    semantic_alignment=2.0
    elf_lora_rank=4
    ;;
  *) exit 2 ;;
esac

epochs=${TRUE_E2E_EPOCHS:-60}
eval_every=${TRUE_E2E_EVAL_EVERY:-1000}
run_tag=qc4wyals_oof5_elfb_${arm}_truee2e_ep${epochs}_seed49_20260908
output_dir=/data/engs-pnpl/glandau/elf-runs/$run_tag

export WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
export WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
export WANDB_RUN_GROUP=qc4wyals_oof5_elfb_truee2e_20260908
export WANDB_RUN_NAME=$run_tag
export WANDB_DIR=${WANDB_DIR:-$data_root/wandb}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-$data_root/wandb-data}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$data_root/wandb-cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$data_root/xdg-cache}
export HF_HOME=${HF_HOME:-$data_root/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TORCH_HOME=${TORCH_HOME:-$data_root/torch}
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=${TMPDIR:-$data_root/tmp}
mkdir -p "$output_dir" "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"

cd "$project_root"
args=(
  scripts/train_npz_meg_to_frozen_elf.py
  --train-npz "$train_npz"
  --val-npz "$val_npz"
  --output_dir "$output_dir"
  --checkpoint_path "$oof_flat_checkpoint"
  --elf-checkpoint-path "$oof_flat_checkpoint"
  --semantic-projector-checkpoint "$oof_flat_checkpoint"
  --semantic-projector-kind flat
  --semantic-projector-dropout 0.05
  --model ELF-B
  --adapter-kind meg2sem_projector
  --meg2sem-checkpoint "$meg2sem_checkpoint"
  --meg2sem-output-normalization always
  --meg-standardization none
  --meg-clip-boundary -1
  --freeze-meg2sem-batchnorm-stats
  --meg2sem-unfreeze-scope "$meg_scope"
  --train-semantic-projector
  --device cuda
  --num-examples 0
  --eval-num-examples 110
  --retrieval-num-examples 110
  --batch_size 8
  --epochs "$epochs"
  --eval_every "$eval_every"
  --retrieval-eval-every "$eval_every"
  --retrieval-batch-size 64
  --target-encode-batch-size 64
  --num_sampling_steps 32
  --train-timestep-mode sampling_schedule
  --train-timestep-steps 32
  --context_length 64
  --semantic-hidden-dim 4096
  --semantic-alignment-loss-weight "$semantic_alignment"
  --semantic-alignment-loss-type cosine
  --semantic-contrastive-loss-weight 0.10
  --semantic-contrastive-temperature 0.07
  --semantic-geometry-loss-weight 0.10
  --semantic-rank-distill-loss-weight 0.10
  --semantic-rank-distill-temperature 0.07
  --decoder-content-token-weight "$content_token"
  --decoder-content-bow-loss-weight "$content_bow"
  --decoder-content-precision-loss-weight "$content_precision"
  --decoder-condition-pairing-margin-weight "$pairing"
  --decoder-condition-pairing-margin 0.10
  --lr "$projector_lr"
  --semantic-projector-lr "$projector_lr"
  --meg2sem-lr "$meg_lr"
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
  --save-eval-checkpoints
  --eval-checkpoint-top-k 1
  --eval-checkpoint-every 0
  --skip-final-checkpoint
  --seed 49
  --use_wandb
  --wandb_project "$WANDB_PROJECT"
  --wandb_entity "$WANDB_ENTITY"
  --wandb_group "$WANDB_RUN_GROUP"
  --wandb_run_name "$WANDB_RUN_NAME"
  --wandb_sample_examples 16
  --wandb_notes "Legitimate true E2E matrix initialized from the full nested-OOF flat ELF-B adapter. Train2652 and val110 only; protected test26 is absent."
)

if [[ "$train_meg2sem" == "1" ]]; then
  args+=(--train-meg2sem)
fi
if [[ "$elf_lora_rank" -gt 0 ]]; then
  args+=(
    --elf-lora-rank "$elf_lora_rank"
    --elf-lora-alpha "$((elf_lora_rank * 2))"
    --elf-lora-dropout 0.05
    --elf-lora-targets attention,mlp
    --elf-lora-last-n-blocks "$elf_lora_last_n"
    --elf-lr "$elf_lr"
  )
fi

exec "$python_bin" "${args[@]}"
