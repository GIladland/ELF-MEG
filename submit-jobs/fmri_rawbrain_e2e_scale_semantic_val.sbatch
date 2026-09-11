#!/usr/bin/env bash
#SBATCH --job-name=fmri_e2e_scale
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-2
#SBATCH --output=submit-jobs/fmri_rawbrain_e2e_scale_semantic_val-%A_%a.out
#SBATCH --error=submit-jobs/fmri_rawbrain_e2e_scale_semantic_val-%A_%a.err

set -euo pipefail

# Brain-side validation grid corresponding to the known-text oracle scale grid.
# It loads only MRI2SEM train_x/val_x and never reads test_x.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
minilm_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
ada_dir=$DATA_ROOT/ada/fmri_knowntext_v1
mri_run=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1
baseline_run=fmri_minilm_rawbrain_e2e_projector_long100_resdelaytok_h1024_l2_h8_storybal_b32_seed49_20260830
baseline_checkpoint=/data/engs-pnpl/glandau/elf-runs/$baseline_run/best.pt

export MODEL=ELF-L
export CHECKPOINT_PATH=embedded-language-flows/ELF-L-owt-torch
export NPZ_PATH=$minilm_dir/oracle_exact_minilm384_train_val.npz
export SEMANTIC_ALIGNMENT_TARGET_NPZ=$NPZ_PATH
export SEMANTIC_ADAPTER_KIND=residual_mlp
export SEMANTIC_MAPPER_HIDDEN_DIM=1024
export INIT_E2E_ADAPTER_MISMATCH=residual-mlp
semantic=minilm384
decoder=elfl
oracle_run=fmri_knowntext_minilm384_elfl_long_ep100_testfirst_k64_seed49_20260830

case "$SLURM_ARRAY_TASK_ID" in
  0)
    ;;
  1)
    semantic=ada002
    decoder=elfb
    oracle_run=fmri_knowntext_ada002_elfb_long_ep300_testfirst_k64_seed49_20260830
    export MODEL=ELF-B
    export CHECKPOINT_PATH=embedded-language-flows/ELF-B-owt-torch
    export NPZ_PATH=$ada_dir/oracle_exact_ada002_train_val.npz
    export SEMANTIC_ALIGNMENT_TARGET_NPZ=$NPZ_PATH
    export SEMANTIC_ADAPTER_KIND=residual_identity
    ;;
  2)
    semantic=ada002
    decoder=elfl
    oracle_run=fmri_knowntext_ada002_elfl_long_ep100_testfirst_k64_seed49_20260830
    export NPZ_PATH=$ada_dir/oracle_exact_ada002_train_val.npz
    export SEMANTIC_ALIGNMENT_TARGET_NPZ=$NPZ_PATH
    export SEMANTIC_ADAPTER_KIND=residual_identity
    ;;
  *) exit 2 ;;
esac

export SOURCE_NPZ=$NPZ_PATH
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export TRAIN_BALANCE_KEY=story
export BRAIN_INPUT_NPZ=$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz
export BRAIN_TRAIN_KEY=train_x
export BRAIN_VAL_KEY=val_x
export BRAIN_MODEL_CHECKPOINT=$baseline_checkpoint
export BRAIN_SEMANTIC_OUTPUT_DIM=1536
export BRAIN_HIDDEN_DIM=2048
export BRAIN_RES_BLOCKS=4
export BRAIN_DROPOUT=0.1
export BRAIN_UNFREEZE=projector
export BRAIN_LR=1e-5

export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings
export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=1.0
export SEMANTIC_ALIGNMENT_LOSS_TYPE=cosine
export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.1
export SEMANTIC_CONTRASTIVE_TEMPERATURE=0.07
export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.1
export RUN_TAG=fmri_rawbrain_e2e_${semantic}_${decoder}_projector_long100_storybal_seed49_20260830${SCALE_RUN_SUFFIX:-}
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$DATA_ROOT/ada/fmri_knowntext_v1/${RUN_TAG}_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/$oracle_run/best.pt
if [[ -n "${INIT_E2E_CHECKPOINT_OVERRIDE:-}" ]]; then
  export INIT_E2E_CHECKPOINT=$INIT_E2E_CHECKPOINT_OVERRIDE
fi

export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export SKIP_ARC_SETUP=1
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=266
export EVAL_NUM_EXAMPLES=266
export RETRIEVAL_NUM_EXAMPLES=266
export EVAL_EVERY=1000
export RETRIEVAL_EVAL_EVERY=1000
export RETRIEVAL_BATCH_SIZE=256
export EPOCHS=100
export BATCH_SIZE=32
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_DELAY_TOKENS=4
export SEMANTIC_DELAY_LAYERS=2
export SEMANTIC_DELAY_HEADS=8
export SEMANTIC_INPUT_PROJECTION_DIM=0
export ADAPTER_DROPOUT=0.1
export NORMALIZE_SEMANTIC_PROJECTION=0
export TRAIN_SEMANTIC_INPUT_ONLY=1
export FREEZE_ELF=1
export LR=1e-4
export LR_SCHEDULE=cosine
export WARMUP_STEPS=1000
export MIN_LR_RATIO=0.01
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CHECKPOINT_METRIC=content_word_overlap
export SAVE_FINAL_CHECKPOINT=1
export SAVE_EVAL_CHECKPOINTS=1
export EVAL_CHECKPOINT_TOP_K=1
export EVAL_CHECKPOINT_EVERY=5
export SEED=49
export USE_WANDB=1
export WANDB_RUN_GROUP=fmri_rawbrain_e2e_scale_semantic_val_20260830
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Validation-only raw-fMRI E2E semantic/decoder scale grid. Train MRI2SEM projector at 1e-5 plus zero-init semantic residual at 1e-4; freeze trunk and oracle decoder. Train11725/val266; test107 brain remains closed. ADA adapts the pretrained 1536-D MiniLM projector toward exact ADA supervision."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
