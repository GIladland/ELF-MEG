#!/usr/bin/env bash
#SBATCH --job-name=elf_mri_ml_x8lora
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-7%4
#SBATCH --output=submit-jobs/train_elf_archscale_mri_ml_x8_lora_search_20260906-%A_%a.out
#SBATCH --error=submit-jobs/train_elf_archscale_mri_ml_x8_lora_search_20260906-%A_%a.err

set -euo pipefail

# Promote the best x8-backed adapter independently for ELF-M and ELF-L, then
# isolate LoRA-only from adapter+LoRA training.  Predicted MRI val266 remains
# selection-only; protected test107 brain vectors are absent.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118/bin/python

task=${SLURM_ARRAY_TASK_ID:-0}
model_index=$((task / 4))
variant_index=$((task % 4))

case "$model_index" in
  0)
    model=ELF-M; model_slug=elfm
    recipes=(x8_const2e5 x8_cos1e5)
    ;;
  1)
    model=ELF-L; model_slug=elfl
    recipes=(x8_const1e5 x8_cos5e6)
    ;;
  *) exit 2 ;;
esac

selector_args=(scripts/select_best_elf_candidate.py --metric word_content_mean --format tsv)
for recipe in "${recipes[@]}"; do
  for adapter_label in resmlp delaytokens; do
    tag=elf_archscale_mri_minilm_x8_${model_slug}_${recipe}_ep4_wcmean_seed7_20260906_${adapter_label}_adapter_val266_wcmean_seed49
    selector_args+=(--candidate "$tag,$model,/data/engs-pnpl/glandau/elf-runs/$tag")
  done
done

cd "$PROJECT_ROOT"
IFS=$'\t' read -r selected_tag selected_model selected_dir selected_checkpoint selected_score < <(
  "$python_bin" "${selector_args[@]}"
)
if [[ "$selected_model" != "$model" || ! -f "$selected_checkpoint" ]]; then
  echo "Invalid promoted candidate: model=$selected_model checkpoint=$selected_checkpoint" >&2
  exit 3
fi

case "$selected_tag" in
  *_resmlp_adapter_*) adapter_kind=residual_mlp ;;
  *_delaytokens_adapter_*) adapter_kind=residual_delay_tokens ;;
  *) echo "Cannot infer MRI adapter kind from $selected_tag" >&2; exit 4 ;;
esac

case "$variant_index" in
  0) arm=loraonly_r4last4; rank=4; last_blocks=4; adapter_lr=0; elf_lr=5e-7 ;;
  1) arm=adapterlora_r4last4; rank=4; last_blocks=4; adapter_lr=2e-6; elf_lr=5e-7 ;;
  2) arm=loraonly_r8all; rank=8; last_blocks=-1; adapter_lr=0; elf_lr=2e-7 ;;
  3) arm=adapterlora_r8all; rank=8; last_blocks=-1; adapter_lr=1e-6; elf_lr=2e-7 ;;
  *) exit 2 ;;
esac

echo "Promoted $selected_tag score=$selected_score checkpoint=$selected_checkpoint"
export LADDER_TASK_ID_OVERRIDE=4
export MODEL_OVERRIDE=$model
export CHECKPOINT_PATH_OVERRIDE=embedded-language-flows/${model}-owt-torch
export ORACLE_TAG_OVERRIDE=$selected_tag
export RUN_TAG_OVERRIDE=${selected_tag}_${arm}_val266_wcmean_seed49
export SEMANTIC_ADAPTER_KIND_OVERRIDE=$adapter_kind
export INIT_E2E_ADAPTER_MISMATCH_OVERRIDE=error
export TRAIN_SEMANTIC_INPUT_ONLY_OVERRIDE=0
export LR_OVERRIDE=$adapter_lr
export ELF_LORA_RANK_OVERRIDE=$rank
export ELF_LORA_ALPHA_OVERRIDE=$((rank * 2))
export ELF_LORA_DROPOUT_OVERRIDE=0.05
export ELF_LORA_TARGETS_OVERRIDE=attention,mlp
export ELF_LORA_LAST_N_BLOCKS_OVERRIDE=$last_blocks
export ELF_LR_OVERRIDE=$elf_lr
export CURRICULUM_SEMANTIC_ONLY_EPOCHS_OVERRIDE=0
export RETRIEVAL_NUM_EXAMPLES_OVERRIDE=266
export BATCH_SIZE_OVERRIDE=16
export EPOCHS_OVERRIDE=80
export EVAL_EVERY_OVERRIDE=1000
export RETRIEVAL_EVAL_EVERY_OVERRIDE=1000
export SAVE_FINAL_CHECKPOINT_OVERRIDE=0
export SAVE_EVAL_CHECKPOINTS_OVERRIDE=0
export CHECKPOINT_METRIC_OVERRIDE=word_content_mean

exec bash "$PROJECT_ROOT/submit-jobs/train_elf_semantic_scaling_adapter_ladder.sbatch"
