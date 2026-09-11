#!/usr/bin/env bash
#SBATCH --job-name=elfm_breval
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --array=0-26%4
#SBATCH --output=submit-jobs/eval_qc4wyals_elfm_breakthrough_multiseed_20260907-%A_%a.out
#SBATCH --error=submit-jobs/eval_qc4wyals_elfm_breakthrough_multiseed_20260907-%A_%a.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
packed=$data_root/tang_themoth_apples_exact_ada002_textmid_seg10_v1/packed
train_npz=$packed/tang_themoth_exact384_ada002_textmid_seg10000_train_dedup_fp16.npz
val_npz=$packed/tang_themoth_exact384_ada002_textmid_seg10000_val_fp16.npz
meg2sem=/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/qc4wyals/checkpoints/tang_cosine0831_ada_clip_drop03init_cos2_aux80_bank512_s82-epoch=48-val_nDCG_epoch=0.4828.ckpt
oracle=/data/engs-pnpl/glandau/elf-runs/elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906/best.pt
root=/data/engs-pnpl/glandau/elf-runs

task=${SLURM_ARRAY_TASK_ID:-0}
model_index=$((task / 3))
seed_index=$((task % 3))
seeds=(49 101 202)
seed=${seeds[$seed_index]}
adapter_kind=meg2sem_projector
lora=0
lora_last=4
case "$model_index" in
  0) arm=postctx_frozenelf_balanced; source=meg_elfm_postctx_frozenelf_balanced_ep80_seed49_20260906 ;;
  1) arm=postctx_lora4_balanced; source=meg_elfm_postctx_lora4last4_balanced_ep80_seed49_20260906; lora=4 ;;
  2) arm=postctx_lora4_strongcontent; source=meg_elfm_postctx_lora4last4_strongcontent_ep80_seed49_20260906; lora=4 ;;
  3) arm=postctx_frozenadapter_lora4_strongcontent; source=meg_elfm_postctx_frozenadapter_lora4last4_strongcontent_ep80_seed49_20260906; lora=4 ;;
  4) arm=postctx_lora8last8_strongcontent; source=meg_elfm_postctx_lora8last8_strongcontent_ep80_seed49_20260906; lora=8; lora_last=8 ;;
  5) arm=postctx_fullmeg_lora4_balanced; source=meg_elfm_postctx_fullmeg_lora4last4_balanced_ep80_seed49_20260906; lora=4 ;;
  6) arm=temporal_frozenelf; source=meg_elfm_temporal64_postctx_frozenelf_strongcontent_ep80_seed49_20260906; adapter_kind=meg_context ;;
  7) arm=temporal_lora4last4; source=meg_elfm_temporal64_postctx_lora4last4_strongcontent_ep80_seed49_20260906; adapter_kind=meg_context; lora=4 ;;
  8) arm=temporal_lora8last8; source=meg_elfm_temporal64_postctx_lora8last8_strongcontent_ep80_seed49_20260906; adapter_kind=meg_context; lora=8; lora_last=8 ;;
  *) exit 2 ;;
esac
selected=$root/$source/best_adapter.pt
[[ -f "$selected" ]] || { echo "Missing selected checkpoint $selected" >&2; exit 2; }
run_tag=eval_meg_elfm_breakthrough_${arm}_val110_seed${seed}_20260907
output_dir=$root/$run_tag
mkdir -p "$output_dir" "$data_root/tmp" "$data_root/wandb" "$data_root/wandb-data"
export WANDB_ENTITY=giladland-university-of-oxford WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=meg_elfm_breakthrough_multiseed_20260907 WANDB_RUN_NAME=$run_tag
export WANDB_DIR=$data_root/wandb HF_HOME=$data_root/huggingface TOKENIZERS_PARALLELISM=false
export WANDB_DATA_DIR=$data_root/wandb-data
export PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=$data_root/tmp
cd "$project_root"
args=(
  scripts/train_npz_meg_to_frozen_elf.py
  --train-npz "$train_npz" --val-npz "$val_npz" --output_dir "$output_dir"
  --checkpoint_path "$oracle" --elf-checkpoint-path "$oracle"
  --semantic-projector-checkpoint "$oracle" --semantic-context-teacher-checkpoint "$oracle"
  --init-e2e-checkpoint "$selected" --model ELF-M --adapter-kind "$adapter_kind"
  --meg-standardization none --meg-clip-boundary -1 --device cuda --eval-only
  --num-examples 0 --eval-num-examples 110 --retrieval-num-examples 110 --batch_size 8
  --retrieval-batch-size 64 --target-encode-batch-size 64 --num_sampling_steps 32
  --train-timestep-mode sampling_schedule --train-timestep-steps 32
  --context_length 64 --semantic-hidden-dim 4096 --cfg_scale 1.0 --self_cond_cfg_scale 1.0
  --generation-t5-retrieval --oracle-semantic-eval --freeze-meg2sem-batchnorm-stats
  --seed "$seed" --use_wandb --wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY"
  --wandb_group "$WANDB_RUN_GROUP" --wandb_run_name "$WANDB_RUN_NAME"
  --wandb_notes "Fixed-seed val110 reevaluation for the finite ELF-M breakthrough matrix. Same candidates/settings; protected test26 absent."
)
if [[ "$adapter_kind" == meg2sem_projector ]]; then
  args+=(--meg2sem-checkpoint "$meg2sem" --meg2sem-output-normalization always)
fi
if [[ "$lora" -gt 0 ]]; then
  args+=(--elf-lora-rank "$lora" --elf-lora-alpha "$((lora * 2))" --elf-lora-dropout 0.05 \
    --elf-lora-targets attention,mlp --elf-lora-last-n-blocks "$lora_last")
fi
"$python_bin" "${args[@]}"

bert_root=$data_root/python-packages/bert-score-0.3.13
export PYTHONPATH=$bert_root${PYTHONPATH:+:$PYTHONPATH}
"$python_bin" scripts/add_bertscore_to_metrics.py \
  --metrics-json "$output_dir/best_metrics.json" \
  --output "$output_dir/best_metrics.bertscore.json" --batch-size 8 --device cuda --top-k 10
