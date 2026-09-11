#!/usr/bin/env bash
#SBATCH --job-name=qc4w_finaleval
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --array=0-4%4
#SBATCH --output=submit-jobs/eval_qc4wyals_selected_true_meg_e2e-%A_%a.out
#SBATCH --error=submit-jobs/eval_qc4wyals_selected_true_meg_e2e-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=${PYTHON_BIN:-$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python}
packed_root=$data_root/tang_themoth_apples_exact_ada002_textmid_seg10_v1/packed
train_npz=$packed_root/tang_themoth_exact384_ada002_textmid_seg10000_train_dedup_fp16.npz
eval_split=${EVAL_SPLIT:-val}
case "$eval_split" in
  val)
    eval_npz=$packed_root/tang_themoth_exact384_ada002_textmid_seg10000_val_fp16.npz
    eval_rows=110
    ;;
  test)
    eval_npz=$packed_root/tang_themoth_exact384_ada002_textmid_seg10000_test_fp16.npz
    eval_rows=26
    ;;
  *)
    echo "EVAL_SPLIT must be val or test, got: $eval_split" >&2
    exit 2
    ;;
esac

: "${SELECTED_CHECKPOINT:?Set SELECTED_CHECKPOINT to one validation-selected E2E checkpoint}"
: "${BASE_ORACLE_CHECKPOINT:?Set BASE_ORACLE_CHECKPOINT to the ELF oracle used to initialize it}"
meg2sem_checkpoint=${MEG2SEM_CHECKPOINT:-/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/qc4wyals/checkpoints/tang_cosine0831_ada_clip_drop03init_cos2_aux80_bank512_s82-epoch=48-val_nDCG_epoch=0.4828.ckpt}

for required in "$python_bin" "$train_npz" "$eval_npz" "$SELECTED_CHECKPOINT" "$BASE_ORACLE_CHECKPOINT" "$meg2sem_checkpoint"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 2; }
done

seeds=(49 101 202 303 404)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
selected_tag=${SELECTED_TAG:-$(basename "$SELECTED_CHECKPOINT" .pt)}
run_tag=eval_qc4wyals_${eval_split}${eval_rows}_${selected_tag}_seed${seed}_20260901
output_dir=/data/engs-pnpl/glandau/elf-runs/$run_tag

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
mkdir -p "$output_dir" "$TMPDIR"

cd "$project_root"
args=(
  scripts/train_npz_meg_to_frozen_elf.py
  --train-npz "$train_npz"
  --val-npz "$eval_npz"
  --output_dir "$output_dir"
  --checkpoint_path "$BASE_ORACLE_CHECKPOINT"
  --elf-checkpoint-path "$BASE_ORACLE_CHECKPOINT"
  --semantic-projector-checkpoint "$BASE_ORACLE_CHECKPOINT"
  --init-e2e-checkpoint "$SELECTED_CHECKPOINT"
  --adapter-kind meg2sem_projector
  --meg2sem-checkpoint "$meg2sem_checkpoint"
  --meg2sem-output-normalization always
  --meg-standardization none
  --meg-clip-boundary -1
  --device cuda
  --eval-only
  --num-examples 0
  --eval-num-examples "$eval_rows"
  --retrieval-num-examples "$eval_rows"
  --batch_size 8
  --retrieval-batch-size 64
  --target-encode-batch-size 64
  --num_sampling_steps 32
  --train-timestep-mode sampling_schedule
  --train-timestep-steps 32
  --context_length 64
  --semantic-hidden-dim 4096
  --cfg_scale 1.0
  --self_cond_cfg_scale 1.0
  --cond_dropout_prob 0.0
  --generation-t5-retrieval
  --oracle-semantic-eval
  --freeze-meg2sem-batchnorm-stats
  --seed "$seed"
)

elf_lora_rank=${ELF_LORA_RANK:-0}
if [[ "$elf_lora_rank" -gt 0 ]]; then
  args+=(
    --elf-lora-rank "$elf_lora_rank"
    --elf-lora-alpha "${ELF_LORA_ALPHA:-16}"
    --elf-lora-dropout "${ELF_LORA_DROPOUT:-0.05}"
    --elf-lora-targets "${ELF_LORA_TARGETS:-attention,mlp}"
    --elf-lora-last-n-blocks "${ELF_LORA_LAST_N_BLOCKS:--1}"
  )
fi

exec "$python_bin" "${args[@]}"
