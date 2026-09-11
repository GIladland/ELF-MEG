#!/usr/bin/env bash
#SBATCH --job-name=elf_arch_x16
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-1%2
#SBATCH --output=submit-jobs/train_elf_architecture_scaling_meg_x16-%A_%a.out
#SBATCH --error=submit-jobs/train_elf_architecture_scaling_meg_x16-%A_%a.err

set -euo pipefail

# Combined capacity/data scaling: continue each converged x1 M/L oracle on the
# anchor-balanced x16 MEG/ADA corpus. Its 86,538 physical rows have median ten
# words (range 7--13), directly reinforcing the desired output-length prior.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) model=ELF-M; model_slug=elfm ;;
  1) model=ELF-L; model_slug=elfl ;;
  *) exit 2 ;;
esac

export RUN_TAG=elf_archscale_meg_ada_x16_${model_slug}_continue_x1_content_seed7_20260904
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export NPZ_PATH=$DATA_ROOT/semantic_scaling_20260902/meg_ada/meg_ada_x16_anchorbalanced_auditfirst.npz
export TARGET_LATENTS_CACHE=$DATA_ROOT/semantic_scaling_20260902/meg_ada/meg_ada_x16_anchorbalanced_auditfirst_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/elf_archscale_meg_ada_x1_${model_slug}_exact_compute44k_content_seed7_20260904/best.pt
export INIT_E2E_ADAPTER_MISMATCH=error
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export MODEL=$model
export CHECKPOINT_PATH=embedded-language-flows/${model}-owt-torch

export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=110
export RETRIEVAL_NUM_EXAMPLES=110
# Batch 16 makes three epochs approximately 16.2k continuation updates.
export EPOCHS=3
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

export LR=2e-5
export ELF_LR=2e-5
export WEIGHT_DECAY=0.0
export LR_SCHEDULE=cosine
export WARMUP_STEPS=250
export MIN_LR_RATIO=0.02
export CHECKPOINT_METRIC=content_word_overlap
export SAVE_FINAL_CHECKPOINT=0
export SAVE_EVAL_CHECKPOINTS=0
export EVAL_CHECKPOINT_TOP_K=1
export EVAL_CHECKPOINT_EVERY=1
export SEED=7

export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=elf_architecture_data_scaling_meg_x16_20260904
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_SAMPLE_EXAMPLES=16
export WANDB_NOTES="Combined ELF model/data scaling. Continue the matched converged x1 $model exact-ADA decoder on the anchor-balanced x16 Tang/Apples corpus. Fixed first 110 known-text audit rows; no MEG-predicted validation vector is loaded. Content-first checkpointing; downstream exact/raw/adapter evaluations are separate."

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
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
  --batch-size 8 --device cuda --top-k 10
