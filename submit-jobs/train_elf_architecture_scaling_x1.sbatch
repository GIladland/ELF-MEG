#!/usr/bin/env bash
#SBATCH --job-name=elf_arch_train
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-3%4
#SBATCH --output=submit-jobs/train_elf_architecture_scaling_x1-%A_%a.out
#SBATCH --error=submit-jobs/train_elf_architecture_scaling_x1-%A_%a.err

set -euo pipefail

# Model-capacity axis at fixed x1 semantic/text data.  The existing ELF-B x1
# checkpoints are the B controls; these four arms fill in ELF-M and ELF-L for
# both modalities.  No brain-predicted validation vector is loaded here.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) modality=mri_minilm; model=ELF-M; model_slug=elfm; epochs=60; audit_n=107 ;;
  1) modality=mri_minilm; model=ELF-L; model_slug=elfl; epochs=60; audit_n=107 ;;
  2) modality=meg_ada; model=ELF-M; model_slug=elfm; epochs=250; audit_n=136 ;;
  3) modality=meg_ada; model=ELF-L; model_slug=elfl; epochs=250; audit_n=136 ;;
  *) exit 2 ;;
esac

if [[ "$modality" == mri_minilm ]]; then
  known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
  export NPZ_PATH=$known_dir/oracle_exact_minilm384_all12098_testfirst.npz
  export TARGET_LATENTS_CACHE=$known_dir/fmri_minilm_knowntext_exact384_all12098_testfirst_full_k64_ep100_b64_seed49_20260829_t5small_latents_fp16.npy
else
  known_dir=$DATA_ROOT/tang_themoth_apples_exact_ada002_textmid_seg10_v1
  export NPZ_PATH=$known_dir/tang_apples_ada002_knowntext_all2788_testfirst.npz
  export TARGET_LATENTS_CACHE=$known_dir/tang_apples_ada002_knowntext_all2788_t5small_latents_fp16.npy
fi

export RUN_TAG=elf_archscale_${modality}_x1_${model_slug}_exact_compute44k_content_seed7_20260904
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export MODEL=$model
export CHECKPOINT_PATH=embedded-language-flows/${model}-owt-torch
export INIT_E2E_CHECKPOINT=
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence

export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=$audit_n
export RETRIEVAL_NUM_EXAMPLES=$audit_n
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

# About 44k updates per arm. Match the successful ELF-B full-memorization
# learning-rate recipe; select content rather than the misleading early
# minimum-WER state observed in the first M/L pilot.
export LR=1e-4
export ELF_LR=1e-4
export WEIGHT_DECAY=0.0
export LR_SCHEDULE=constant
export WARMUP_STEPS=0
export MIN_LR_RATIO=1.0
export CHECKPOINT_METRIC=content_word_overlap
export SAVE_FINAL_CHECKPOINT=0
export SAVE_EVAL_CHECKPOINTS=0
export EVAL_CHECKPOINT_TOP_K=1
export EVAL_CHECKPOINT_EVERY=1
export SEED=7

export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=elf_architecture_scaling_x1_20260903
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_SAMPLE_EXAMPLES=16
export WANDB_NOTES="Corrected ELF architecture-capacity scaling at fixed x1 corpus. Fresh stock $model plus a fresh semantic adapter; about 44k optimization updates with the successful ELF-B full-memorization LR. Exact semantic/text pairs only, no brain-predicted validation vectors. Content-first checkpointing avoids the early short-output minimum-WER failure seen in the first M/L pilot."

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
