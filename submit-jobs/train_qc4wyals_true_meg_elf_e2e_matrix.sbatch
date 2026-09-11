#!/usr/bin/env bash
#SBATCH --job-name=qc4w_truee2e
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-7%3
#SBATCH --output=submit-jobs/train_qc4wyals_true_meg_elf_e2e_matrix-%A_%a.out
#SBATCH --error=submit-jobs/train_qc4wyals_true_meg_elf_e2e_matrix-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=${PYTHON_BIN:-$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python}
packed_root=$data_root/tang_themoth_apples_exact_ada002_textmid_seg10_v1/packed
train_npz=$packed_root/tang_themoth_exact384_ada002_textmid_seg10000_train_dedup_fp16.npz
val_npz=$packed_root/tang_themoth_exact384_ada002_textmid_seg10000_val_fp16.npz
meg2sem_checkpoint=${MEG2SEM_CHECKPOINT:-/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/qc4wyals/checkpoints/tang_cosine0831_ada_clip_drop03init_cos2_aux80_bank512_s82-epoch=48-val_nDCG_epoch=0.4828.ckpt}
oracle_checkpoint=${ORACLE_CHECKPOINT:-/data/engs-pnpl/glandau/elf-runs/tang_apples_exact_ada002_fresh_elfb_full_memorize_all2788_ep1000_seed7_20260901/checkpoints/eval_step_00035000_score_0.773228.pt}

for required in "$python_bin" "$train_npz" "$val_npz" "$meg2sem_checkpoint" "$oracle_checkpoint"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 2
  fi
done

train_meg2sem=0
train_projector=1
unfreeze_elf=0
last_n_blocks=0
elf_lora_rank=0
elf_lora_alpha=16
elf_lora_last_n=-1
meg_lr=2e-7
meg_scope=all
projector_lr=5e-6
elf_lr=1e-6
semantic_alignment=0.0

case "$SLURM_ARRAY_TASK_ID" in
  0)
    arm=projector_only_frozen_meg_frozen_elf
    ;;
  1)
    arm=projector_lora4_last4_frozen_meg
    elf_lora_rank=4
    elf_lora_alpha=8
    elf_lora_last_n=4
    ;;
  2)
    arm=projector_lora8_all_frozen_meg
    elf_lora_rank=8
    elf_lora_alpha=16
    ;;
  3)
    arm=fullmeg_projector_frozen_elf
    train_meg2sem=1
    semantic_alignment=0.5
    ;;
  4)
    arm=fullmeg_projector_lora4_last4
    train_meg2sem=1
    semantic_alignment=0.5
    elf_lora_rank=4
    elf_lora_alpha=8
    elf_lora_last_n=4
    ;;
  5)
    arm=fullmeg_projector_last2elf
    train_meg2sem=1
    semantic_alignment=0.5
    unfreeze_elf=1
    last_n_blocks=2
    ;;
  6)
    arm=outputmeg_projector_frozen_elf_strongalign
    train_meg2sem=1
    meg_scope=output
    meg_lr=5e-8
    semantic_alignment=2.0
    ;;
  7)
    arm=outputmeg_projector_lora4_last4_strongalign
    train_meg2sem=1
    meg_scope=output
    meg_lr=5e-8
    semantic_alignment=2.0
    elf_lora_rank=4
    elf_lora_alpha=8
    elf_lora_last_n=4
    ;;
  *) exit 2 ;;
esac

epochs=${TRUE_E2E_EPOCHS:-30}
eval_every=${TRUE_E2E_EVAL_EVERY:-1000}
suffix=${TRUE_E2E_SUFFIX:-}
if [[ -n "$suffix" ]]; then
  suffix=_$suffix
fi
run_tag=tang_apples_qc4wyals_true_meg_${arm}_ep${epochs}${suffix}_seed49_20260901
output_dir=/data/engs-pnpl/glandau/elf-runs/$run_tag

export WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
export WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
export WANDB_RUN_GROUP=tang_apples_qc4wyals_true_meg_e2e_20260901
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
  --checkpoint_path "$oracle_checkpoint"
  --elf-checkpoint-path "$oracle_checkpoint"
  --semantic-projector-checkpoint "$oracle_checkpoint"
  --adapter-kind meg2sem_projector
  --meg2sem-checkpoint "$meg2sem_checkpoint"
  --meg2sem-output-normalization always
  --meg-standardization none
  --meg-clip-boundary -1
  --device cuda
  --num-examples 0
  --eval-num-examples 110
  --retrieval-num-examples 110
  --batch_size "${TRUE_E2E_BATCH_SIZE:-8}"
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
  --lr "$projector_lr"
  --meg2sem-lr "$meg_lr"
  --meg2sem-unfreeze-scope "$meg_scope"
  --freeze-meg2sem-batchnorm-stats
  --semantic-projector-lr "$projector_lr"
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
  --save-eval-checkpoints
  --eval-checkpoint-top-k 1
  --eval-checkpoint-every 0
  --seed 49
  --use_wandb
  --wandb_project "$WANDB_PROJECT"
  --wandb_entity "$WANDB_ENTITY"
  --wandb_group "$WANDB_RUN_GROUP"
  --wandb_run_name "$WANDB_RUN_NAME"
  --wandb_sample_examples 16
  --wandb_notes "True raw-MEG E2E qc4wyals-to-ELF. Fit 2652 train rows and select on val110. Protected BirthOfANation test26 MEG is absent. Exact ADA/text may be known to ELF by explicit memorization contract."
)

if [[ "$train_meg2sem" == "1" ]]; then
  args+=(--train-meg2sem)
fi
if [[ "$train_projector" == "1" ]]; then
  args+=(--train-semantic-projector)
fi
if [[ "$elf_lora_rank" -gt 0 ]]; then
  args+=(
    --elf-lora-rank "$elf_lora_rank"
    --elf-lora-alpha "$elf_lora_alpha"
    --elf-lora-dropout 0.05
    --elf-lora-targets attention,mlp
    --elf-lora-last-n-blocks "$elf_lora_last_n"
    --elf-lr "$elf_lr"
  )
fi
if [[ "$unfreeze_elf" == "1" ]]; then
  args+=(--unfreeze-elf --unfreeze-last-n-blocks "$last_n_blocks" --elf-lr "$elf_lr")
fi

exec "$python_bin" "${args[@]}"
