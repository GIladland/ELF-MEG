#!/usr/bin/env bash
#SBATCH --job-name=fmri_oracle_scale
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-2
#SBATCH --output=submit-jobs/fmri_knowntext_oracle_scale_semantic-%A_%a.out
#SBATCH --error=submit-jobs/fmri_knowntext_oracle_scale_semantic-%A_%a.err

set -euo pipefail

# Known-text oracle grid. It intentionally includes the 107 target sentences,
# but never loads their fMRI vectors; its purpose is decoder/interface capacity.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
minilm_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
ada_dir=$DATA_ROOT/ada/fmri_knowntext_v1

export MODEL=ELF-L
export CHECKPOINT_PATH=embedded-language-flows/ELF-L-owt-torch
export NPZ_PATH=$minilm_dir/oracle_exact_minilm384_all12098_testfirst.npz
export BATCH_SIZE=16
export EPOCHS=100
export LR=1e-5
semantic=minilm384
decoder=elfl

case "$SLURM_ARRAY_TASK_ID" in
  0)
    ;;
  1)
    semantic=ada002
    decoder=elfb
    export MODEL=ELF-B
    export CHECKPOINT_PATH=embedded-language-flows/ELF-B-owt-torch
    export NPZ_PATH=$ada_dir/oracle_exact_ada002_all12098_testfirst.npz
    export BATCH_SIZE=64
    export EPOCHS=300
    export LR=2e-5
    ;;
  2)
    semantic=ada002
    decoder=elfl
    export NPZ_PATH=$ada_dir/oracle_exact_ada002_all12098_testfirst.npz
    ;;
  *) exit 2 ;;
esac

export SOURCE_NPZ=$NPZ_PATH
export RUN_TAG=fmri_knowntext_${semantic}_${decoder}_long_ep${EPOCHS}_testfirst_k64_seed49_20260830
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$DATA_ROOT/ada/fmri_knowntext_v1/${RUN_TAG}_t5small_latents_fp16.npy
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export SKIP_ARC_SETUP=1

export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=107
export RETRIEVAL_NUM_EXAMPLES=107
export EVAL_EVERY=1000
export RETRIEVAL_EVAL_EVERY=1000
export RETRIEVAL_BATCH_SIZE=256
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_ADAPTER_KIND=flat
export SEMANTIC_INPUT_PROJECTION_DIM=0
export ADAPTER_DROPOUT=0.0
export FREEZE_ELF=0
export LAST_N_BLOCKS=-1
export LR_SCHEDULE=cosine
export WARMUP_STEPS=1000
export MIN_LR_RATIO=0.01
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_PROB=0.5
export DECODER_NOISE_SCALE=2.5
export TRAIN_TARGET_MASK_MODE=full
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CHECKPOINT_METRIC=content_word_overlap
export SAVE_FINAL_CHECKPOINT=1
export SAVE_EVAL_CHECKPOINTS=1
export EVAL_CHECKPOINT_TOP_K=1
export EVAL_CHECKPOINT_EVERY=5
export SEED=49
export USE_WANDB=1
export WANDB_RUN_GROUP=fmri_knowntext_oracle_scale_semantic_20260830
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Known-text decoder/interface capacity grid. All 12,098 exact target sentences, including target107 text, are intentionally available; no test brain vector is used. Compare MiniLM/ADA and ELF-B/ELF-L before brain-side validation."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
