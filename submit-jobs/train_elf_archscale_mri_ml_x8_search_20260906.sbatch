#!/usr/bin/env bash
#SBATCH --job-name=elf_mri_ml_x8
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-3%4
#SBATCH --output=submit-jobs/train_elf_archscale_mri_ml_x8_search_20260906-%A_%a.out
#SBATCH --error=submit-jobs/train_elf_archscale_mri_ml_x8_search_20260906-%A_%a.err

set -euo pipefail

# High-capacity, high-data MiniLM stage. Each recovered x1 M/L oracle is
# continued on the anchor-balanced x8 corpus: 96,784 unique pairs and 181,736
# physical rows. No MRI-predicted vector is loaded.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    model=ELF-M; model_slug=elfm; parent_recipe=continue_const8e5; parent_epochs=180
    recipe=x8_const2e5; lr=2e-5; schedule=constant; warmup=0; min_ratio=1.0
    ;;
  1)
    model=ELF-M; model_slug=elfm; parent_recipe=continue_cos5e5; parent_epochs=180
    recipe=x8_cos1e5; lr=1e-5; schedule=cosine; warmup=500; min_ratio=0.05
    ;;
  2)
    model=ELF-L; model_slug=elfl; parent_recipe=fresh_const2e5; parent_epochs=130
    recipe=x8_const1e5; lr=1e-5; schedule=constant; warmup=0; min_ratio=1.0
    ;;
  3)
    model=ELF-L; model_slug=elfl; parent_recipe=fresh_cos1e5; parent_epochs=130
    recipe=x8_cos5e6; lr=5e-6; schedule=cosine; warmup=500; min_ratio=0.05
    ;;
  *) exit 2 ;;
esac

parent_tag=elf_archscale_mri_minilm_x1_${model_slug}_${parent_recipe}_ep${parent_epochs}_wcmean_seed7_20260906
init=/data/engs-pnpl/glandau/elf-runs/$parent_tag/best.pt
if [[ ! -f "$init" ]]; then
  echo "Missing recovered x1 initialization checkpoint: $init" >&2
  exit 3
fi

epochs=4
export RUN_TAG=elf_archscale_mri_minilm_x8_${model_slug}_${recipe}_ep${epochs}_wcmean_seed7_20260906
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export NPZ_PATH=$DATA_ROOT/semantic_scaling_20260902/mri_minilm/mri_minilm_x8_anchorbalanced_auditfirst.npz
export TARGET_LATENTS_CACHE=$DATA_ROOT/semantic_scaling_20260902/mri_minilm/mri_minilm_x8_anchorbalanced_auditfirst_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=$init
export INIT_E2E_ADAPTER_MISMATCH=error
export MODEL=$model
export CHECKPOINT_PATH=embedded-language-flows/${model}-owt-torch
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence

export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=266
export RETRIEVAL_NUM_EXAMPLES=266
export EPOCHS=$epochs
export BATCH_SIZE=16
export EVAL_EVERY=1000
export RETRIEVAL_EVAL_EVERY=1000
export RETRIEVAL_BATCH_SIZE=128

export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_ADAPTER_KIND=flat
export ADAPTER_DROPOUT=0.0
export NORMALIZE_SEMANTIC_PROJECTION=0
export FREEZE_ELF=0
export LAST_N_BLOCKS=-1
export ELF_LORA_RANK=0

export TRAIN_TARGET_MASK_MODE=full
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CFG_SCALE=1.0
export SELF_COND_CFG_SCALE=1.0
export COND_DROPOUT_PROB=0.0
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_CONTENT_TOKEN_WEIGHT=1.0
export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.0
export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.0
export DECODER_NOISE_SCALE=2.5
export GENERATION_T5_RETRIEVAL=1

export LR=$lr
export ELF_LR=$lr
export WEIGHT_DECAY=0.0
export LR_SCHEDULE=$schedule
export WARMUP_STEPS=$warmup
export MIN_LR_RATIO=$min_ratio
export CHECKPOINT_METRIC=word_content_mean
export SAVE_FINAL_CHECKPOINT=0
export SAVE_EVAL_CHECKPOINTS=0
export EVAL_CHECKPOINT_TOP_K=1
export SEED=7

export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=elf_archscale_mri_ml_x8_20260906
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_SAMPLE_EXAMPLES=16
export WANDB_NOTES="High-capacity/high-data MiniLM oracle. Continue recovered $model on x8: 96,784 unique pairs, 181,736 physical rows, 50% exact-anchor replay. Optimize mean word/content F1 on fixed known-text audit; no MRI-predicted vector is loaded."

export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"

python_bin=$TMP_ENV_ROOT/bin/python
bert_root=$DATA_ROOT/python-packages/bert-score-0.3.13
export PYTHONNOUSERSITE=1
export PYTHONPATH=$bert_root${PYTHONPATH:+:$PYTHONPATH}
"$python_bin" "$PROJECT_ROOT/scripts/add_bertscore_to_metrics.py" \
  --metrics-json "$OUTPUT_DIR/best_metrics.json" \
  --output "$OUTPUT_DIR/best_metrics.bertscore.json" \
  --batch-size 8 --device cuda --top-k 10
