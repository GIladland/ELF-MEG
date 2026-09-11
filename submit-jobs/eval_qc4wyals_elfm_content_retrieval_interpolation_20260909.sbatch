#!/usr/bin/env bash
#SBATCH --job-name=meg_cr_interp
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --array=0-8%3
#SBATCH --output=submit-jobs/eval_qc4wyals_elfm_content_retrieval_interpolation_20260909-%A_%a.out
#SBATCH --error=submit-jobs/eval_qc4wyals_elfm_content_retrieval_interpolation_20260909-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
runs_root=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
packed=$data_root/tang_themoth_apples_exact_ada002_textmid_seg10_v1/packed
train_npz=$packed/tang_themoth_exact384_ada002_textmid_seg10000_train_dedup_fp16.npz
val_npz=$packed/tang_themoth_exact384_ada002_textmid_seg10000_val_fp16.npz
meg2sem=/data/engs-pnpl/glandau/MEG2SEM/MEG2SEM/wandb/qc4wyals/checkpoints/tang_cosine0831_ada_clip_drop03init_cos2_aux80_bank512_s82-epoch=48-val_nDCG_epoch=0.4828.ckpt
base_projector=$runs_root/elf_archscale_meg_ada_x16_elfm_continue_const1e5_ep8_wcmean_seed7_20260906_resid_adapter_val110_wcmean_seed49/best.pt
content_checkpoint=$runs_root/elf_archscale_meg_ada_x16_elfm_continue_const1e5_ep8_wcmean_seed7_20260906_resid_adapter_val110_wcmean_seed49_truee2e_output_projector_lora4last4_val110_wcmean_seed49/best_adapter.pt
retrieval_checkpoint=$runs_root/elf_archscale_meg_ada_x16_elfm_continue_const1e5_ep8_wcmean_seed7_20260906_resid_adapter_val110_wcmean_seed49_truee2e_allmeg_projector_lora4last4_val110_wcmean_seed49/best_adapter.pt

alphas=(0.000 0.125 0.250 0.375 0.500 0.625 0.750 0.875 1.000)
alpha=${alphas[${SLURM_ARRAY_TASK_ID:-0}]}
alpha_tag=${alpha/./p}
run_tag=meg_elfm_content_to_retrieval_weightinterp_a${alpha_tag}_val110_seed49_20260909
output_dir=$runs_root/$run_tag
tmp_root=${SLURM_TMPDIR:-$data_root/tmp}
interpolated=$tmp_root/${run_tag}.pt

for required in "$python_bin" "$train_npz" "$val_npz" "$meg2sem" "$base_projector" "$content_checkpoint" "$retrieval_checkpoint"; do
  [[ -f "$required" ]] || { echo "Missing $required" >&2; exit 2; }
done
mkdir -p "$output_dir" "$tmp_root" "$data_root/wandb" "$data_root/wandb-data"
export WANDB_ENTITY=giladland-university-of-oxford WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=meg_elfm_content_retrieval_interpolation_20260909 WANDB_RUN_NAME=$run_tag
export WANDB_DIR=$data_root/wandb WANDB_DATA_DIR=$data_root/wandb-data
export HF_HOME=$data_root/huggingface TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=$tmp_root

cd "$project_root"
"$python_bin" scripts/interpolate_e2e_checkpoints.py \
  --baseline "$content_checkpoint" --candidate "$retrieval_checkpoint" \
  --alpha "$alpha" --interpolate-model --output "$interpolated"

"$python_bin" scripts/train_npz_meg_to_frozen_elf.py \
  --train-npz "$train_npz" --val-npz "$val_npz" --output_dir "$output_dir" \
  --checkpoint_path "$base_projector" --elf-checkpoint-path "$base_projector" \
  --semantic-projector-checkpoint "$base_projector" \
  --semantic-projector-kind residual_identity --semantic-projector-mapper-hidden-dim 1024 \
  --init-e2e-checkpoint "$interpolated" --model ELF-M --adapter-kind meg2sem_projector \
  --meg2sem-checkpoint "$meg2sem" --meg2sem-output-normalization always \
  --meg-standardization none --meg-clip-boundary -1 --freeze-meg2sem-batchnorm-stats \
  --elf-lora-rank 4 --elf-lora-alpha 8 --elf-lora-dropout 0.05 \
  --elf-lora-targets attention,mlp --elf-lora-last-n-blocks 4 \
  --device cuda --eval-only --num-examples 0 --eval-num-examples 110 \
  --retrieval-num-examples 110 --batch_size 8 --retrieval-batch-size 64 \
  --target-encode-batch-size 64 --num_sampling_steps 32 --context_length 64 \
  --semantic-hidden-dim 4096 --cfg_scale 1.0 --self_cond_cfg_scale 1.0 \
  --generation-t5-retrieval --oracle-semantic-eval --seed 49 --use_wandb \
  --wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY" \
  --wandb_group "$WANDB_RUN_GROUP" --wandb_run_name "$WANDB_RUN_NAME" \
  --wandb_notes "Validation-only full-checkpoint interpolation: alpha=0 content specialist, alpha=1 retrieval specialist. Same val110/110 options; test26 absent."

rm -f "$interpolated"
