#!/usr/bin/env bash
#SBATCH --job-name=ta_adat5ctx
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-4%3
#SBATCH --output=submit-jobs/train_tang_apples_ada_t5_context_matrix-%A_%a.out
#SBATCH --error=submit-jobs/train_tang_apples_ada_t5_context_matrix-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
source_dir=$data_root/tang_themoth_apples_exact_ada002_textmid_seg10_v1
source_npz=$source_dir/tang_apples_ada002_knowntext_all2788_testfirst.npz
latent_cache=$source_dir/tang_apples_ada002_knowntext_all2788_t5small_latents_fp16.npy
teacher_npz=$source_dir/tang_apples_ada002_knowntext_all2788_t5context.npz
env_root=$data_root/conda-envs/braindiffusion-elf-torch271cu118

bridge_type=resampler
input_key=ada_token
input_mask_key=ada_token_mask
context_length=27
bridge_layers=2
freeze_args=(--freeze-elf)
case "$SLURM_ARRAY_TASK_ID" in
  0) arm=ada_resamp2_ctx27_frozen ;;
  1) arm=ada_resamp4_ctx27_frozen; bridge_layers=4 ;;
  2) arm=ada_resamp2_ctx64_frozen; context_length=64 ;;
  3) arm=ada_resamp2_ctx27_last4; freeze_args=(--last_n_blocks 4) ;;
  4)
    arm=teacher_orderpreserve_frozen
    bridge_type=order_preserving
    input_key=target_t5_latents
    input_mask_key=t5_attention_mask
    freeze_args=(--freeze-elf)
    ;;
  *) exit 2 ;;
esac

if ! command -v conda >/dev/null 2>&1; then
  source /apps/system/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$env_root"
export XDG_CACHE_HOME=$data_root/xdg-cache
export HF_HOME=$data_root/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TOKENIZERS_PARALLELISM=false
export TMPDIR=$data_root/tmp
export WANDB_DIR=$data_root/wandb
export WANDB_DATA_DIR=$data_root/wandb-data
export WANDB_CACHE_DIR=$data_root/wandb-cache
mkdir -p "$TMPDIR" "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR"
cd "$project_root"

if [[ ! -f "$teacher_npz" ]]; then
  python scripts/export_text_t5_context_npz.py \
    --source-npz "$source_npz" \
    --target-latents-cache "$latent_cache" \
    --output "$teacher_npz"
fi

run_suffix=${T5_CONTEXT_SUFFIX:-}
if [[ -n "$run_suffix" ]]; then
  run_suffix=_$run_suffix
fi
output_dir=/data/engs-pnpl/glandau/elf-runs/tang_apples_${arm}${run_suffix}_seed49_20260901
python scripts/train_npz_t5_bridge_to_elf.py \
  --npz-path "$teacher_npz" \
  --input-key "$input_key" \
  --input-mask-key "$input_mask_key" \
  --target-latents-key target_t5_latents \
  --target-mask-key t5_attention_mask \
  --target-ids-key t5_input_ids \
  --sentence-key sentence \
  --bridge-type "$bridge_type" \
  --bridge-layers "$bridge_layers" \
  --bridge-heads 8 \
  --bridge-dropout 0.05 \
  --context_length "$context_length" \
  --batch_size 64 \
  --steps "${T5_CONTEXT_STEPS:-8000}" \
  --lr 1e-4 \
  --eval_every "${T5_CONTEXT_EVAL_EVERY:-1000}" \
  --retrieval-eval-every "${T5_CONTEXT_EVAL_EVERY:-1000}" \
  --eval-num-examples 26 \
  --retrieval-num-examples 26 \
  --retrieval-batch-size 128 \
  --num_sampling_steps 32 \
  --train-timestep-steps 32 \
  --cfg_scale 1.0 \
  --self_cond_cfg_scale 1.0 \
  --cond_dropout_prob 0.0 \
  --seed 49 \
  --device cuda \
  --use_wandb \
  --wandb_project BrainDiffusion \
  --wandb_entity giladland-university-of-oxford \
  --wandb_group tang_apples_ada_t5_context_20260901 \
  --wandb_run_name tang_apples_${arm}${run_suffix}_seed49_20260901 \
  --wandb_notes "Known-text ADA/T5-context diffusion oracle. All exact ADA/text rows may train; no qc4wyals MEG-predicted test vector is loaded." \
  --output_dir "$output_dir" \
  "${freeze_args[@]}"
