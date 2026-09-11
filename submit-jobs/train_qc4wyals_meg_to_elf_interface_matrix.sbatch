#!/usr/bin/env bash
#SBATCH --job-name=qc4w_megelf
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-18%4
#SBATCH --output=submit-jobs/train_qc4wyals_meg_to_elf_interface_matrix-%A_%a.out
#SBATCH --error=submit-jobs/train_qc4wyals_meg_to_elf_interface_matrix-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
interface_dir=$DATA_ROOT/tang_themoth_qc4wyals_ada_v2/interface

skip_oracle_init=${SKIP_ORACLE_INIT:-0}
if [[ "$skip_oracle_init" == 1 ]]; then
  # Ablation: start from the stock pretrained ELF checkpoint and jointly learn
  # the semantic adapter/ELF weights on the selected leakage-safe interface
  # pack, instead of inheriting the deliberately memorized exact-ADA oracle.
  oracle_checkpoint=
else
  : "${ORACLE_RUN:?Set ORACLE_RUN to the selected exact-ADA ELF run directory or best.pt path}"
  if [[ "$ORACLE_RUN" == *.pt ]]; then
    oracle_checkpoint=$ORACLE_RUN
  else
    oracle_checkpoint=$ORACLE_RUN/best.pt
  fi
  if [[ ! -f "$oracle_checkpoint" ]]; then
    echo "Missing oracle checkpoint: $oracle_checkpoint" >&2
    exit 2
  fi
fi

export SOURCE_NPZ=$interface_dir/qc4wyals_predicted_train_valtail_ada002.npz
export SEMANTIC_ALIGNMENT_TARGET_NPZ=$interface_dir/qc4wyals_exact_alignment_train_valtail_ada002.npz
export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings
export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=0.10
export SEMANTIC_ALIGNMENT_LOSS_TYPE=cosine
export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.02
export SEMANTIC_CONTRASTIVE_TEMPERATURE=0.07
export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.02
export SEMANTIC_CONTENT_LOSS_WEIGHT=0.0
export CURRICULUM_SEMANTIC_ONLY_EPOCHS=0
export CURRICULUM_DENOISER_LOSS_WEIGHT=0.10
export CURRICULUM_DECODER_LOSS_WEIGHT=0.10
export SEMANTIC_ADAPTER_KIND=flat
export SEMANTIC_MAPPER_HIDDEN_DIM=512
export INIT_E2E_ADAPTER_MISMATCH=error
export TRAIN_SEMANTIC_INPUT_ONLY=0
export FREEZE_ELF=1
export LAST_N_BLOCKS=-1
export ELF_LORA_RANK=0
export ELF_LORA_ALPHA=16
export ELF_LORA_DROPOUT=0.05
export ELF_LORA_TARGETS=attention,mlp
export ELF_LORA_LAST_N_BLOCKS=-1
export LR=2e-5
export ELF_LR=0
export EPOCHS=${INTERFACE_EPOCHS:-150}

case "$SLURM_ARRAY_TASK_ID" in
  0)
    arm=pred_adapter
    ;;
  1)
    arm=pred_residual_identity_inputonly
    # ADA is one 1536-D vector, not four delayed 384-D blocks.  Preserve the
    # oracle projector and learn a zero-output 1536 -> 1536 residual correction.
    export SEMANTIC_ADAPTER_KIND=residual_identity
    export INIT_E2E_ADAPTER_MISMATCH=direct-identity
    export TRAIN_SEMANTIC_INPUT_ONLY=1
    export LR=1e-4
    ;;
  2)
    arm=pred_lora8_adapter
    export ELF_LORA_RANK=8
    export ELF_LR=5e-6
    ;;
  3)
    arm=pred_last2_adapter
    export FREEZE_ELF=0
    export LAST_N_BLOCKS=2
    export ELF_LR=3e-6
    ;;
  4)
    arm=mixed_exactall_predtrain_adapter
    export SOURCE_NPZ=$interface_dir/qc4wyals_mixed_exactall_predtrain_valtail_ada002.npz
    export SEMANTIC_ALIGNMENT_TARGET_NPZ=$interface_dir/qc4wyals_mixed_exact_alignment_valtail_ada002.npz
    export EPOCHS=${INTERFACE_EPOCHS:-100}
    ;;
  5)
    arm=robust_residualaug_adapter
    export SOURCE_NPZ=$interface_dir/qc4wyals_robust_exactall_predtrain_residualaug_valtail_ada002.npz
    export SEMANTIC_ALIGNMENT_TARGET_NPZ=$interface_dir/qc4wyals_robust_exact_alignment_valtail_ada002.npz
    export EPOCHS=${INTERFACE_EPOCHS:-60}
    ;;
  6)
    arm=robust_residualaug_lora8
    export SOURCE_NPZ=$interface_dir/qc4wyals_robust_exactall_predtrain_residualaug_valtail_ada002.npz
    export SEMANTIC_ALIGNMENT_TARGET_NPZ=$interface_dir/qc4wyals_robust_exact_alignment_valtail_ada002.npz
    export ELF_LORA_RANK=8
    export ELF_LR=5e-6
    export EPOCHS=${INTERFACE_EPOCHS:-60}
    ;;
  7)
    arm=pred_residual_identity_curriculum
    export SEMANTIC_ADAPTER_KIND=residual_identity
    export INIT_E2E_ADAPTER_MISMATCH=direct-identity
    export TRAIN_SEMANTIC_INPUT_ONLY=1
    export LR=1e-4
    export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=0.50
    export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.05
    export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.10
    export CURRICULUM_SEMANTIC_ONLY_EPOCHS=20
    ;;
  8)
    arm=pred_full_elf_adapter
    export FREEZE_ELF=0
    export LAST_N_BLOCKS=-1
    export ELF_LR=1e-6
    export EPOCHS=${INTERFACE_EPOCHS:-150}
    ;;
  9)
    arm=mixed_exactall_predtrain_full_elf_adapter
    export SOURCE_NPZ=$interface_dir/qc4wyals_mixed_exactall_predtrain_valtail_ada002.npz
    export SEMANTIC_ALIGNMENT_TARGET_NPZ=$interface_dir/qc4wyals_mixed_exact_alignment_valtail_ada002.npz
    export FREEZE_ELF=0
    export LAST_N_BLOCKS=-1
    export ELF_LR=5e-7
    export EPOCHS=${INTERFACE_EPOCHS:-100}
    ;;
  10)
    arm=robust_residualaug_full_elf_adapter
    export SOURCE_NPZ=$interface_dir/qc4wyals_robust_exactall_predtrain_residualaug_valtail_ada002.npz
    export SEMANTIC_ALIGNMENT_TARGET_NPZ=$interface_dir/qc4wyals_robust_exact_alignment_valtail_ada002.npz
    export FREEZE_ELF=0
    export LAST_N_BLOCKS=-1
    export ELF_LR=5e-7
    export EPOCHS=${INTERFACE_EPOCHS:-60}
    ;;
  11)
    arm=pred_residual_identity_curriculum_lora8_strongcontent
    export SEMANTIC_ADAPTER_KIND=residual_identity
    export INIT_E2E_ADAPTER_MISMATCH=direct-identity
    export LR=1e-4
    export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=0.50
    export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.05
    export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.10
    export CURRICULUM_SEMANTIC_ONLY_EPOCHS=20
    export ELF_LORA_RANK=8
    export ELF_LR=5e-6
    export DECODER_CONTENT_TOKEN_WEIGHT=4.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.25
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.10
    ;;
  12)
    arm=pred_lora8_adapter_strongcontent
    export ELF_LORA_RANK=8
    export ELF_LR=5e-6
    export DECODER_CONTENT_TOKEN_WEIGHT=4.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.25
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.10
    ;;
  13)
    arm=mixed_exactall_predtrain_lora8_strongcontent
    export SOURCE_NPZ=$interface_dir/qc4wyals_mixed_exactall_predtrain_valtail_ada002.npz
    export SEMANTIC_ALIGNMENT_TARGET_NPZ=$interface_dir/qc4wyals_mixed_exact_alignment_valtail_ada002.npz
    export ELF_LORA_RANK=8
    export ELF_LR=5e-6
    export EPOCHS=${INTERFACE_EPOCHS:-100}
    export DECODER_CONTENT_TOKEN_WEIGHT=4.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.25
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.10
    ;;
  14)
    # Continue the strongest robust flat-context checkpoint conservatively.
    # For a flat adapter, project_semantic() is exactly the raw ADA vector, so
    # semantic-vector retrieval is invariant; the context loss directly trains
    # the part that controls diffusion-latent retrieval.
    arm=robust_residualaug_context_preserve
    export SOURCE_NPZ=$interface_dir/qc4wyals_robust_exactall_predtrain_residualaug_valtail_ada002.npz
    export SEMANTIC_ALIGNMENT_TARGET_NPZ=$interface_dir/qc4wyals_robust_exact_alignment_valtail_ada002.npz
    export LR=5e-6
    export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=0.0
    export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.0
    export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.20
    export EPOCHS=${INTERFACE_EPOCHS:-60}
    ;;
  15)
    # Joint context/reconstruction continuation with stronger lexical losses.
    arm=robust_residualaug_context_content
    export SOURCE_NPZ=$interface_dir/qc4wyals_robust_exactall_predtrain_residualaug_valtail_ada002.npz
    export SEMANTIC_ALIGNMENT_TARGET_NPZ=$interface_dir/qc4wyals_robust_exact_alignment_valtail_ada002.npz
    export LR=1e-5
    export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=0.0
    export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.0
    export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.10
    export DECODER_CONTENT_TOKEN_WEIGHT=4.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.25
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.10
    export DECODER_CONDITION_PAIRING_MARGIN_WEIGHT=0.10
    export EPOCHS=${INTERFACE_EPOCHS:-60}
    ;;
  16)
    # Preserve the validated context adapter exactly and adapt only a small
    # diffusion-side LoRA. This tests generation gains without changing the
    # raw semantic interface or the learned context projection.
    arm=robust_residualaug_frozenadapter_lora4
    export SOURCE_NPZ=$interface_dir/qc4wyals_robust_exactall_predtrain_residualaug_valtail_ada002.npz
    export SEMANTIC_ALIGNMENT_TARGET_NPZ=$interface_dir/qc4wyals_robust_exact_alignment_valtail_ada002.npz
    export FREEZE_SEMANTIC_ADAPTER=1
    export ELF_LORA_RANK=4
    export ELF_LR=1e-6
    export LR=1e-6
    export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=0.0
    export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.0
    export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.0
    export EPOCHS=${INTERFACE_EPOCHS:-60}
    ;;
  17)
    # Very limited ELF unfreezing from the validated robust interface.
    arm=robust_residualaug_frozenadapter_last2
    export SOURCE_NPZ=$interface_dir/qc4wyals_robust_exactall_predtrain_residualaug_valtail_ada002.npz
    export SEMANTIC_ALIGNMENT_TARGET_NPZ=$interface_dir/qc4wyals_robust_exact_alignment_valtail_ada002.npz
    export FREEZE_SEMANTIC_ADAPTER=1
    export FREEZE_ELF=0
    export LAST_N_BLOCKS=2
    export ELF_LR=5e-7
    export LR=1e-6
    export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=0.0
    export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.0
    export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.0
    export EPOCHS=${INTERFACE_EPOCHS:-60}
    ;;
  18)
    # Small joint update: context adapter plus rank-4 LoRA at conservative rates.
    arm=robust_residualaug_context_lora4
    export SOURCE_NPZ=$interface_dir/qc4wyals_robust_exactall_predtrain_residualaug_valtail_ada002.npz
    export SEMANTIC_ALIGNMENT_TARGET_NPZ=$interface_dir/qc4wyals_robust_exact_alignment_valtail_ada002.npz
    export LR=5e-6
    export ELF_LORA_RANK=4
    export ELF_LR=1e-6
    export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=0.0
    export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.0
    export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.10
    export EPOCHS=${INTERFACE_EPOCHS:-60}
    ;;
  *) exit 2 ;;
esac

# Optional leakage-safe transformed train/validation pack.  This permits the
# same architecture matrix to be rerun after a mapper selected on val110,
# while retaining the exact-ADA alignment target and protected-test boundary.
if [[ -n "${INTERFACE_SOURCE_OVERRIDE:-}" ]]; then
  export SOURCE_NPZ=$INTERFACE_SOURCE_OVERRIDE
fi

interface_suffix=${INTERFACE_SUFFIX:-}
if [[ -n "$interface_suffix" ]]; then
  interface_suffix=_$interface_suffix
fi
oracle_tag=${ORACLE_TAG:-$(basename "${oracle_checkpoint%/best.pt}")}
if [[ -z "$oracle_tag" ]]; then
  oracle_tag=stock_elfb
fi
export RUN_TAG=tang_apples_qc4wyals_${arm}_from_${oracle_tag}${interface_suffix}_seed49_20260901
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export NPZ_PATH=$SOURCE_NPZ
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export MODEL=ELF-B
export CHECKPOINT_PATH=embedded-language-flows/ELF-B-owt-torch
export INIT_E2E_CHECKPOINT=$oracle_checkpoint
export SEMANTIC_INPUT_PROJECTION_DIM=0
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=${ORACLE_SEMANTIC_HIDDEN_DIM:-4096}
export ADAPTER_DROPOUT=0.05
export NORMALIZE_SEMANTIC_PROJECTION=0
# Allow staged runs to lock a validation-selected adapter while adapting only
# LoRA/ELF weights.  The default remains joint adapter training.
export FREEZE_SEMANTIC_ADAPTER=${FREEZE_SEMANTIC_ADAPTER:-0}
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=110
export EVAL_NUM_EXAMPLES=110
export RETRIEVAL_NUM_EXAMPLES=110
export GENERATION_T5_RETRIEVAL=1
export BATCH_SIZE=64
export EVAL_EVERY=${INTERFACE_EVAL_EVERY:-1000}
export RETRIEVAL_EVAL_EVERY=${INTERFACE_EVAL_EVERY:-1000}
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export LR_SCHEDULE=cosine
export WARMUP_STEPS=250
export MIN_LR_RATIO=0.01
export DECODER_LOSS_WEIGHT=1.0
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_PROB=0.5
export DECODER_NOISE_SCALE=2.5
export DECODER_CONTENT_TOKEN_WEIGHT=${DECODER_CONTENT_TOKEN_WEIGHT:-2.0}
export DECODER_CONTENT_BOW_LOSS_WEIGHT=${DECODER_CONTENT_BOW_LOSS_WEIGHT:-0.10}
export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=${DECODER_CONTENT_PRECISION_LOSS_WEIGHT:-0.05}
export DECODER_CONDITION_PAIRING_MARGIN_WEIGHT=${DECODER_CONDITION_PAIRING_MARGIN_WEIGHT:-0.05}
export DECODER_CONDITION_PAIRING_MARGIN=${DECODER_CONDITION_PAIRING_MARGIN:-0.10}
export TRAIN_TARGET_MASK_MODE=full
export CHECKPOINT_METRIC=content_word_overlap
export SAVE_FINAL_CHECKPOINT=1
export SAVE_EVAL_CHECKPOINTS=1
export EVAL_CHECKPOINT_TOP_K=1
export EVAL_CHECKPOINT_EVERY=0
export SEED=49
export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=tang_apples_qc4wyals_meg_to_elf_interface_20260901
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_SAMPLE_EXAMPLES=16
export TARGET_LATENTS_CACHE=$interface_dir/${arm}_t5small_latents_fp16.npy
export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1
export WANDB_NOTES="MEG-to-ELF interface tuning from validation-selected qc4wyals. Train brain predictions=2652; selection=held-out val110; protected BirthOfANation test26 predictions are absent. Exact text/ADA rows may include all known texts by explicit known-text diffusion contract."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
