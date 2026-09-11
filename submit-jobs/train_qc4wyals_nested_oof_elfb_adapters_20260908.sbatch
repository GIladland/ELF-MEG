#!/usr/bin/env bash
#SBATCH --job-name=qc4_oofelfb
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-1%2
#SBATCH --output=submit-jobs/train_qc4wyals_nested_oof_elfb_adapters_20260908-%A_%a.out
#SBATCH --error=submit-jobs/train_qc4wyals_nested_oof_elfb_adapters_20260908-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
interface=${OOF_INTERFACE:-$DATA_ROOT/tang_themoth_qc4wyals_nested_oof_20260908/interface}
oracle=/data/engs-pnpl/glandau/elf-runs/tang_apples_exact_ada002_fresh_elfb_full_memorize_all2788_ep1000_seed7_20260901/best.pt

export SOURCE_NPZ=$interface/qc4wyals_oof_predicted_train_valtail_ada002.npz
export SEMANTIC_ALIGNMENT_TARGET_NPZ=$interface/qc4wyals_oof_exact_alignment_train_valtail_ada002.npz
export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings
export NPZ_PATH=$SOURCE_NPZ
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export MODEL=ELF-B
export CHECKPOINT_PATH=embedded-language-flows/ELF-B-owt-torch
export INIT_E2E_CHECKPOINT=$oracle
export SEMANTIC_INPUT_PROJECTION_DIM=0
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export ADAPTER_DROPOUT=0.05
export NORMALIZE_SEMANTIC_PROJECTION=0
export FREEZE_ELF=1
export LAST_N_BLOCKS=-1
export ELF_LORA_RANK=0
export ELF_LR=0
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=110
export EVAL_NUM_EXAMPLES=110
export RETRIEVAL_NUM_EXAMPLES=110
export GENERATION_T5_RETRIEVAL=1
export BATCH_SIZE=64
export EPOCHS=${OOF_ADAPTER_EPOCHS:-300}
export EVAL_EVERY=${OOF_ADAPTER_EVAL_EVERY:-500}
export RETRIEVAL_EVAL_EVERY=${OOF_ADAPTER_EVAL_EVERY:-500}
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export LR_SCHEDULE=cosine
export WARMUP_STEPS=250
export MIN_LR_RATIO=0.01
export DECODER_LOSS_WEIGHT=1.0
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_PROB=0.5
export DECODER_NOISE_SCALE=2.5
export DECODER_CONTENT_TOKEN_WEIGHT=2.0
export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.10
export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.05
export DECODER_CONDITION_PAIRING_MARGIN_WEIGHT=0.05
export DECODER_CONDITION_PAIRING_MARGIN=0.10
export TRAIN_TARGET_MASK_MODE=full
export CURRICULUM_SEMANTIC_ONLY_EPOCHS=0
export CURRICULUM_DENOISER_LOSS_WEIGHT=0.10
export CURRICULUM_DECODER_LOSS_WEIGHT=0.10
export CHECKPOINT_METRIC=content_word_overlap
export SAVE_FINAL_CHECKPOINT=1
export SAVE_EVAL_CHECKPOINTS=1
export EVAL_CHECKPOINT_TOP_K=1
export EVAL_CHECKPOINT_EVERY=0
export SEED=49

case "$SLURM_ARRAY_TASK_ID" in
  0)
    arm=flat_context
    export SEMANTIC_ADAPTER_KIND=flat
    export INIT_E2E_ADAPTER_MISMATCH=error
    export TRAIN_SEMANTIC_INPUT_ONLY=0
    export FREEZE_SEMANTIC_ADAPTER=0
    export LR=1e-5
    export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=0.0
    export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.0
    export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.20
    ;;
  1)
    arm=residual_identity_joint
    export SEMANTIC_ADAPTER_KIND=residual_identity
    export INIT_E2E_ADAPTER_MISMATCH=direct-identity
    export TRAIN_SEMANTIC_INPUT_ONLY=0
    export FREEZE_SEMANTIC_ADAPTER=0
    export LR=1e-4
    export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=0.50
    export SEMANTIC_ALIGNMENT_LOSS_TYPE=cosine
    export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.05
    export SEMANTIC_CONTRASTIVE_TEMPERATURE=0.07
    export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.10
    export CURRICULUM_SEMANTIC_ONLY_EPOCHS=20
    ;;
  *) exit 2 ;;
esac

for path in "$SOURCE_NPZ" "$SEMANTIC_ALIGNMENT_TARGET_NPZ" "$oracle"; do
  [[ -f "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done

export RUN_TAG=${OOF_RUN_PREFIX:-qc4wyals_nested_oof5_elfb}_${arm}_ep300_seed49_20260908
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$interface/${arm}_t5small_latents_fp16.npy
export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1
export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=${OOF_WANDB_GROUP:-qc4wyals_nested_oof_elfb_gap_20260908}
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_SAMPLE_EXAMPLES=20
export WANDB_NOTES=${OOF_WANDB_NOTES:-"ELF-B adapter trained on 2,652 nested session-OOF MEG predictions. Each train prediction comes from a scratch MEG2SEM that excluded its outer sessions; checkpoint selection used a separate inner session. Validation110 is tail-only and protected test26 is absent."}

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
