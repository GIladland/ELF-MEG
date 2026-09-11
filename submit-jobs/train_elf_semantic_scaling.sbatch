#!/usr/bin/env bash
#SBATCH --job-name=elf_scale_train
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-3%4
#SBATCH --output=submit-jobs/train_elf_semantic_scaling-%A_%a.out
#SBATCH --error=submit-jobs/train_elf_semantic_scaling-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
scale_root=$DATA_ROOT/semantic_scaling_20260902

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    modality=mri_minilm; scale=x4; epochs=8; audit_n=266
    init=/data/engs-pnpl/glandau/elf-runs/fmri_exact_minilm384_fresh_elfb_full_memorize_all12098_ep600_seed7_20260901/best.pt
    ;;
  1)
    modality=mri_minilm; scale=x8; epochs=4; audit_n=266
    init=/data/engs-pnpl/glandau/elf-runs/fmri_exact_minilm384_fresh_elfb_full_memorize_all12098_ep600_seed7_20260901/best.pt
    ;;
  2)
    modality=meg_ada; scale=x4; epochs=40; audit_n=110
    init=/data/engs-pnpl/glandau/elf-runs/tang_apples_exact_ada002_fresh_elfb_full_memorize_all2788_ep1000_seed7_20260901/best.pt
    ;;
  3)
    modality=meg_ada; scale=x16; epochs=9; audit_n=110
    init=/data/engs-pnpl/glandau/elf-runs/tang_apples_exact_ada002_fresh_elfb_full_memorize_all2788_ep1000_seed7_20260901/best.pt
    ;;
  *) exit 2 ;;
esac

export RUN_TAG=elf_semantic_scaling_${modality}_${scale}_anchorbalanced_compute12k_continuex1_seed7_20260902
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export NPZ_PATH=$scale_root/$modality/${modality}_${scale}_anchorbalanced_auditfirst.npz
export TARGET_LATENTS_CACHE=$scale_root/$modality/${modality}_${scale}_anchorbalanced_auditfirst_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=$init
export INIT_E2E_ADAPTER_MISMATCH=error
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export MODEL=ELF-B
export CHECKPOINT_PATH=embedded-language-flows/ELF-B-owt-torch

export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=$audit_n
export RETRIEVAL_NUM_EXAMPLES=$audit_n
export EPOCHS=${EPOCHS_OVERRIDE:-$epochs}
export BATCH_SIZE=${BATCH_SIZE_OVERRIDE:-64}
export EVAL_EVERY=1000
export RETRIEVAL_EVAL_EVERY=1000
export RETRIEVAL_BATCH_SIZE=256

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

# Conservative continuation from the converged x1 decoder. Every arm gets
# approximately 11k--12k updates in stage 1. Validation is measured every
# 1,000 updates; arms that are still improving at the end are continued from
# their best checkpoint rather than being truncated by a wall-time limit.
export LR=2e-5
export ELF_LR=2e-5
export WEIGHT_DECAY=0.0
export LR_SCHEDULE=cosine
export WARMUP_STEPS=250
export MIN_LR_RATIO=0.02

export CHECKPOINT_METRIC=negative_wer
export SAVE_FINAL_CHECKPOINT=1
export SAVE_EVAL_CHECKPOINTS=1
export EVAL_CHECKPOINT_TOP_K=5
export EVAL_CHECKPOINT_EVERY=1
export SEED=7

# Helpful on shared 48 GB RTX8000 nodes, where fragmentation from another
# allocation can otherwise make an otherwise small batch fail spuriously.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=elf_semantic_scaling_20260902
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_SAMPLE_EXAMPLES=16
export WANDB_NOTES="Stage-1 compute-matched semantic-corpus scaling continuation (about 11k--12k updates). Original exact rows are anchor replayed at about 50 percent; synthetic phrasebank rows add same-corpus coverage. First $audit_n exact known-text rows are the fixed audit. No brain vectors are loaded. Continue only if the terminal validation curve is still improving. Downstream raw-brain and trained-adapter ladders are separate jobs."

export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1

bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"

python_bin=$TMP_ENV_ROOT/bin/python
bert_root=$DATA_ROOT/python-packages/bert-score-0.3.13
export PYTHONNOUSERSITE=1
export PYTHONPATH=$bert_root${PYTHONPATH:+:$PYTHONPATH}
"$python_bin" "$PROJECT_ROOT/scripts/add_bertscore_to_metrics.py" \
  --metrics-json "$OUTPUT_DIR/best_metrics.json" \
  --output "$OUTPUT_DIR/best_metrics.bertscore.json" \
  --batch-size 16 \
  --device cuda \
  --top-k 10
