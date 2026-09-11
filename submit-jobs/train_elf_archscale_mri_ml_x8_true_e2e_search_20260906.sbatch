#!/usr/bin/env bash
#SBATCH --job-name=elf_mri_ml_x8e2e
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-7%4
#SBATCH --output=submit-jobs/train_elf_archscale_mri_ml_x8_true_e2e_search_20260906-%A_%a.out
#SBATCH --error=submit-jobs/train_elf_archscale_mri_ml_x8_true_e2e_search_20260906-%A_%a.err

set -euo pipefail

# True raw-fMRI E2E on the promoted x8-backed M/L adapter.  Compare projector
# versus last-residual-block MRI2SEM updates, with and without final-block ELF
# LoRA.  The native MRI2SEM checkpoint initializes only the brain model; the
# promoted checkpoint initializes the high-capacity ELF and semantic adapter.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118/bin/python
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
mri_run=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1

task=${SLURM_ARRAY_TASK_ID:-0}
model_index=$((task / 4))
variant_index=$((task % 4))

case "$model_index" in
  0) model=ELF-M; model_slug=elfm; recipes=(x8_const2e5 x8_cos1e5) ;;
  1) model=ELF-L; model_slug=elfl; recipes=(x8_const1e5 x8_cos5e6) ;;
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

export FREEZE_ELF=1
export ELF_LORA_RANK=0
export ELF_LORA_ALPHA=8
export ELF_LORA_DROPOUT=0.05
export ELF_LORA_TARGETS=attention,mlp
export ELF_LORA_LAST_N_BLOCKS=4
export ELF_LR=0
case "$variant_index" in
  0) arm=projector_adapter_frozenelf; brain_scope=projector; brain_lr=2e-7 ;;
  1) arm=lastblock_adapter_frozenelf; brain_scope=last_block; brain_lr=5e-8 ;;
  2)
    arm=projector_adapter_lora4last4; brain_scope=projector; brain_lr=2e-7
    export ELF_LORA_RANK=4; export ELF_LR=2e-7
    ;;
  3)
    arm=lastblock_adapter_lora4last4; brain_scope=last_block; brain_lr=5e-8
    export ELF_LORA_RANK=4; export ELF_LR=2e-7
    ;;
  *) exit 2 ;;
esac

for required in "$selected_checkpoint" "$mri_run/model.pt" \
  "$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz" \
  "$known_dir/oracle_exact_minilm384_train_val.npz"; do
  if [[ ! -f "$required" ]]; then echo "Missing required path: $required" >&2; exit 5; fi
done

echo "Promoted $selected_tag score=$selected_score checkpoint=$selected_checkpoint"
export RUN_TAG=${selected_tag}_truee2e_${arm}_val266_wcmean_seed49
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export NPZ_PATH=$known_dir/oracle_exact_minilm384_train_val.npz
export SOURCE_NPZ=$NPZ_PATH
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export TRAIN_BALANCE_KEY=story

export BRAIN_INPUT_NPZ=$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz
export BRAIN_TRAIN_KEY=train_x
export BRAIN_VAL_KEY=val_x
export BRAIN_MODEL_CHECKPOINT=$mri_run/model.pt
export BRAIN_SEMANTIC_OUTPUT_DIM=1536
export BRAIN_PROJECTOR_MISMATCH=error
export BRAIN_HIDDEN_DIM=2048
export BRAIN_RES_BLOCKS=4
export BRAIN_DROPOUT=0.1
export BRAIN_UNFREEZE=$brain_scope
export BRAIN_LR=$brain_lr

export MODEL=$model
export CHECKPOINT_PATH=embedded-language-flows/${model}-owt-torch
export INIT_E2E_CHECKPOINT=$selected_checkpoint
export INIT_E2E_ADAPTER_MISMATCH=error
export SEMANTIC_ADAPTER_KIND=$adapter_kind
export SEMANTIC_MAPPER_HIDDEN_DIM=512
export SEMANTIC_DELAY_TOKENS=4
export SEMANTIC_DELAY_LAYERS=2
export SEMANTIC_DELAY_HEADS=8
export ADAPTER_DROPOUT=0.0
export TRAIN_SEMANTIC_INPUT_ONLY=0
export NORMALIZE_SEMANTIC_PROJECTION=0

export SEMANTIC_ALIGNMENT_TARGET_NPZ=$known_dir/oracle_exact_minilm384_train_val.npz
export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings
export SEMANTIC_ALIGNMENT_LOSS_TYPE=cosine
export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=1.0
export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.1
export SEMANTIC_CONTRASTIVE_TEMPERATURE=0.07
export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.1
export SEMANTIC_RANK_DISTILL_LOSS_WEIGHT=0.5
export SEMANTIC_RANK_DISTILL_TEMPERATURE=0.07
export SEMANTIC_RAW_ANCHOR_LOSS_WEIGHT=1.0
export SEMANTIC_GEOMETRY_LOSS_WEIGHT=1.0

export TARGET_LATENTS_CACHE=$known_dir/oracle_exact_minilm384_trainval_current10_t5small_latents_fp16.npy
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=266
export EVAL_NUM_EXAMPLES=266
export RETRIEVAL_NUM_EXAMPLES=266
export BATCH_SIZE=16
export EPOCHS=100
export EVAL_EVERY=1000
export RETRIEVAL_EVAL_EVERY=1000
export RETRIEVAL_BATCH_SIZE=128
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export LR=1e-6
export WEIGHT_DECAY=0.001
export LR_SCHEDULE=cosine
export WARMUP_STEPS=500
export MIN_LR_RATIO=0.01
export COND_DROPOUT_PROB=0.0
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_CONTENT_TOKEN_WEIGHT=2.0
export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.5
export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.1
export DECODER_CONDITION_PAIRING_MARGIN_WEIGHT=0.25
export DECODER_CONDITION_PAIRING_MARGIN=0.1
export CURRICULUM_SEMANTIC_ONLY_EPOCHS=2
export CURRICULUM_DENOISER_LOSS_WEIGHT=0.1
export CURRICULUM_DECODER_LOSS_WEIGHT=0.1
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export GENERATION_T5_RETRIEVAL=1
export CHECKPOINT_METRIC=word_content_mean
export SAVE_FINAL_CHECKPOINT=0
export SAVE_EVAL_CHECKPOINTS=0
export EVAL_CHECKPOINT_TOP_K=1
export SEED=49

export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=elf_archscale_mri_ml_x8_true_e2e_20260906
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_SAMPLE_EXAMPLES=16
export WANDB_NOTES="True raw-fMRI E2E from native MRI2SEM plus promoted x8 $model adapter. Arm=$arm. Train11725/val266 only; protected test107 brain is absent. Selection=mean word/content F1."
export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"

bert_root=$DATA_ROOT/python-packages/bert-score-0.3.13
export PYTHONNOUSERSITE=1
export PYTHONPATH=$bert_root${PYTHONPATH:+:$PYTHONPATH}
"$python_bin" scripts/add_bertscore_to_metrics.py \
  --metrics-json "$OUTPUT_DIR/best_metrics.json" \
  --output "$OUTPUT_DIR/best_metrics.bertscore.json" \
  --batch-size 8 --device cuda --top-k 10
