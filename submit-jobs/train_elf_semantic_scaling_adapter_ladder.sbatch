#!/usr/bin/env bash
#SBATCH --job-name=elf_scale_adapt
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-5%4
#SBATCH --output=submit-jobs/train_elf_semantic_scaling_adapter_ladder-%A_%a.out
#SBATCH --error=submit-jobs/train_elf_semantic_scaling_adapter_ladder-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1

task_id=${LADDER_TASK_ID_OVERRIDE:-${SLURM_ARRAY_TASK_ID:-0}}
case "$task_id" in
  0) modality=mri_minilm; scale=x4; audit_n=266 ;;
  1) modality=mri_minilm; scale=x8; audit_n=266 ;;
  2) modality=meg_ada; scale=x4; audit_n=110 ;;
  3) modality=meg_ada; scale=x16; audit_n=110 ;;
  4) modality=mri_minilm; scale=x1; audit_n=266 ;;
  5) modality=meg_ada; scale=x1; audit_n=110 ;;
  *) exit 2 ;;
esac

if [[ "$scale" == x1 && "$modality" == mri_minilm ]]; then
  oracle_tag=fmri_exact_minilm384_fresh_elfb_full_memorize_all12098_ep600_seed7_20260901
elif [[ "$scale" == x1 ]]; then
  oracle_tag=tang_apples_exact_ada002_fresh_elfb_full_memorize_all2788_ep1000_seed7_20260901
else
  oracle_tag=elf_semantic_scaling_${modality}_${scale}_anchorbalanced_compute12k_continuex1_seed7_20260902
fi
oracle_tag=${ORACLE_TAG_OVERRIDE:-$oracle_tag}
oracle=/data/engs-pnpl/glandau/elf-runs/$oracle_tag/best.pt
if [[ ! -f "$oracle" ]]; then
  echo "Missing scaled ELF checkpoint: $oracle" >&2
  exit 3
fi

export RUN_TAG=${RUN_TAG_OVERRIDE:-elf_semantic_scaling_${modality}_${scale}_trained_adapter_val${audit_n}_seed49_20260902}
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export INIT_E2E_CHECKPOINT=$oracle
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export MODEL=${MODEL_OVERRIDE:-ELF-B}
export CHECKPOINT_PATH=${CHECKPOINT_PATH_OVERRIDE:-embedded-language-flows/ELF-B-owt-torch}
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=$audit_n
export EVAL_NUM_EXAMPLES=$audit_n
export RETRIEVAL_NUM_EXAMPLES=${RETRIEVAL_NUM_EXAMPLES_OVERRIDE:-$audit_n}
export BATCH_SIZE=${BATCH_SIZE_OVERRIDE:-32}
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export FREEZE_ELF=${FREEZE_ELF_OVERRIDE:-1}
export LAST_N_BLOCKS=-1
export ELF_LORA_RANK=${ELF_LORA_RANK_OVERRIDE:-0}
export ELF_LORA_ALPHA=${ELF_LORA_ALPHA_OVERRIDE:-16}
export ELF_LORA_DROPOUT=${ELF_LORA_DROPOUT_OVERRIDE:-0.05}
export ELF_LORA_TARGETS=${ELF_LORA_TARGETS_OVERRIDE:-attention,mlp}
export ELF_LORA_LAST_N_BLOCKS=${ELF_LORA_LAST_N_BLOCKS_OVERRIDE:--1}
export ELF_LR=${ELF_LR_OVERRIDE:-0}
export ADAPTER_DROPOUT=0.0
export NORMALIZE_SEMANTIC_PROJECTION=0
export TRAIN_TARGET_MASK_MODE=full
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CFG_SCALE=1.0
export SELF_COND_CFG_SCALE=1.0
export COND_DROPOUT_PROB=${COND_DROPOUT_PROB_OVERRIDE:-0.0}
export DENOISER_LOSS_WEIGHT=${DENOISER_LOSS_WEIGHT_OVERRIDE:-1.0}
export DECODER_LOSS_WEIGHT=${DECODER_LOSS_WEIGHT_OVERRIDE:-1.0}
export GENERATION_T5_RETRIEVAL=1
export EVAL_EVERY=${EVAL_EVERY_OVERRIDE:-500}
export RETRIEVAL_EVAL_EVERY=${RETRIEVAL_EVAL_EVERY_OVERRIDE:-500}
export RETRIEVAL_BATCH_SIZE=256
export LR_SCHEDULE=cosine
export WARMUP_STEPS=250
export MIN_LR_RATIO=0.02
export WEIGHT_DECAY=0.0
export CHECKPOINT_METRIC=${CHECKPOINT_METRIC_OVERRIDE:-content_word_overlap}
export SAVE_FINAL_CHECKPOINT=${SAVE_FINAL_CHECKPOINT_OVERRIDE:-1}
export SAVE_EVAL_CHECKPOINTS=${SAVE_EVAL_CHECKPOINTS_OVERRIDE:-1}
export EVAL_CHECKPOINT_TOP_K=${EVAL_CHECKPOINT_TOP_K_OVERRIDE:-5}
export EVAL_CHECKPOINT_EVERY=${EVAL_CHECKPOINT_EVERY_OVERRIDE:-1}
export SEED=49

if [[ "$modality" == mri_minilm ]]; then
  export NPZ_PATH=$DATA_ROOT/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz
  export SOURCE_NPZ=$NPZ_PATH
  export SEMANTIC_ALIGNMENT_TARGET_NPZ=$known_dir/oofmix_exact_minilm384_targets_train23450_val266.npz
  export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings
  export TARGET_LATENTS_CACHE=$known_dir/oofmix_current10_t5small_latents_fp16.npy
  export SEMANTIC_ADAPTER_KIND=${SEMANTIC_ADAPTER_KIND_OVERRIDE:-residual_mlp}
  export SEMANTIC_MAPPER_HIDDEN_DIM=512
  export SEMANTIC_DELAY_TOKENS=4
  export INIT_E2E_ADAPTER_MISMATCH=${INIT_E2E_ADAPTER_MISMATCH_OVERRIDE:-residual-mlp}
  export TRAIN_SEMANTIC_INPUT_ONLY=${TRAIN_SEMANTIC_INPUT_ONLY_OVERRIDE:-1}
  export LR=${LR_OVERRIDE:-1e-5}
  export EPOCHS=${EPOCHS_OVERRIDE:-20}
  export SEMANTIC_ALIGNMENT_LOSS_TYPE=cosine
  export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=${SEMANTIC_ALIGNMENT_LOSS_WEIGHT_OVERRIDE:-1.0}
  export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=${SEMANTIC_CONTRASTIVE_LOSS_WEIGHT_OVERRIDE:-0.1}
  export SEMANTIC_CONTRASTIVE_TEMPERATURE=0.07
  export SEMANTIC_CONTEXT_LOSS_WEIGHT=${SEMANTIC_CONTEXT_LOSS_WEIGHT_OVERRIDE:-0.1}
  export SEMANTIC_RANK_DISTILL_LOSS_WEIGHT=${SEMANTIC_RANK_DISTILL_LOSS_WEIGHT_OVERRIDE:-0.5}
  export SEMANTIC_RANK_DISTILL_TEMPERATURE=0.07
  export SEMANTIC_RAW_ANCHOR_LOSS_WEIGHT=${SEMANTIC_RAW_ANCHOR_LOSS_WEIGHT_OVERRIDE:-1.0}
  export SEMANTIC_GEOMETRY_LOSS_WEIGHT=${SEMANTIC_GEOMETRY_LOSS_WEIGHT_OVERRIDE:-1.0}
  export DECODER_CONTENT_TOKEN_WEIGHT=${DECODER_CONTENT_TOKEN_WEIGHT_OVERRIDE:-2.0}
  export DECODER_CONTENT_BOW_LOSS_WEIGHT=${DECODER_CONTENT_BOW_LOSS_WEIGHT_OVERRIDE:-0.5}
  export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=${DECODER_CONTENT_PRECISION_LOSS_WEIGHT_OVERRIDE:-0.1}
  export DECODER_CONDITION_PAIRING_MARGIN_WEIGHT=${DECODER_CONDITION_PAIRING_MARGIN_WEIGHT_OVERRIDE:-0.25}
  export DECODER_CONDITION_PAIRING_MARGIN=0.1
  export CURRICULUM_SEMANTIC_ONLY_EPOCHS=${CURRICULUM_SEMANTIC_ONLY_EPOCHS_OVERRIDE:-2}
  export CURRICULUM_DENOISER_LOSS_WEIGHT=${CURRICULUM_DENOISER_LOSS_WEIGHT_OVERRIDE:-0.1}
  export CURRICULUM_DECODER_LOSS_WEIGHT=${CURRICULUM_DECODER_LOSS_WEIGHT_OVERRIDE:-0.1}
else
  interface=$DATA_ROOT/tang_themoth_qc4wyals_ada_v2/interface
  export NPZ_PATH=$interface/qc4wyals_robust_exactall_predtrain_residualaug_valtail_ada002.npz
  export SOURCE_NPZ=$NPZ_PATH
  export SEMANTIC_ALIGNMENT_TARGET_NPZ=$interface/qc4wyals_robust_exact_alignment_valtail_ada002.npz
  export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings
  export TARGET_LATENTS_CACHE=$interface/robust_residualaug_adapter_t5small_latents_fp16.npy
  export SEMANTIC_ADAPTER_KIND=${SEMANTIC_ADAPTER_KIND_OVERRIDE:-flat}
  export SEMANTIC_MAPPER_HIDDEN_DIM=${SEMANTIC_MAPPER_HIDDEN_DIM_OVERRIDE:-512}
  export INIT_E2E_ADAPTER_MISMATCH=${INIT_E2E_ADAPTER_MISMATCH_OVERRIDE:-error}
  # Reproduce the validated MEG winner: train the complete flat context
  # adapter, not only its input projection.
  export TRAIN_SEMANTIC_INPUT_ONLY=${TRAIN_SEMANTIC_INPUT_ONLY_OVERRIDE:-0}
  export ADAPTER_DROPOUT=0.05
  export BATCH_SIZE=${BATCH_SIZE_OVERRIDE:-64}
  export EVAL_EVERY=${EVAL_EVERY_OVERRIDE:-250}
  export RETRIEVAL_EVAL_EVERY=${RETRIEVAL_EVAL_EVERY_OVERRIDE:-250}
  export MIN_LR_RATIO=0.01
  export EVAL_CHECKPOINT_TOP_K=1
  export LR=${LR_OVERRIDE:-2e-5}
  export EPOCHS=${EPOCHS_OVERRIDE:-60}
  export SEMANTIC_ALIGNMENT_LOSS_TYPE=cosine
  export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=${SEMANTIC_ALIGNMENT_LOSS_WEIGHT_OVERRIDE:-0.10}
  export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=${SEMANTIC_CONTRASTIVE_LOSS_WEIGHT_OVERRIDE:-0.02}
  export SEMANTIC_CONTRASTIVE_TEMPERATURE=0.07
  export SEMANTIC_CONTEXT_LOSS_WEIGHT=${SEMANTIC_CONTEXT_LOSS_WEIGHT_OVERRIDE:-0.02}
  export SEMANTIC_RAW_ANCHOR_LOSS_WEIGHT=${SEMANTIC_RAW_ANCHOR_LOSS_WEIGHT_OVERRIDE:-0.0}
  export SEMANTIC_GEOMETRY_LOSS_WEIGHT=${SEMANTIC_GEOMETRY_LOSS_WEIGHT_OVERRIDE:-0.0}
  export SEMANTIC_RANK_DISTILL_LOSS_WEIGHT=${SEMANTIC_RANK_DISTILL_LOSS_WEIGHT_OVERRIDE:-0.0}
  export DECODER_CONTENT_TOKEN_WEIGHT=${DECODER_CONTENT_TOKEN_WEIGHT_OVERRIDE:-2.0}
  export DECODER_CONTENT_BOW_LOSS_WEIGHT=${DECODER_CONTENT_BOW_LOSS_WEIGHT_OVERRIDE:-0.10}
  export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=${DECODER_CONTENT_PRECISION_LOSS_WEIGHT_OVERRIDE:-0.05}
  export DECODER_CONDITION_PAIRING_MARGIN_WEIGHT=${DECODER_CONDITION_PAIRING_MARGIN_WEIGHT_OVERRIDE:-0.05}
  export DECODER_CONDITION_PAIRING_MARGIN=0.1
  export CURRICULUM_SEMANTIC_ONLY_EPOCHS=${CURRICULUM_SEMANTIC_ONLY_EPOCHS_OVERRIDE:-0}
fi

export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=elf_semantic_scaling_20260902
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_SAMPLE_EXAMPLES=16
export WANDB_NOTES="Trained-adapter rung for the fixed semantic-corpus scaling ladder. Scaled ELF is frozen. Training uses train/OOF brain predictions plus exact targets; selection uses val$audit_n. MRI test107 and MEG test26 brain vectors are absent."
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
  --device "${BERT_DEVICE:-cuda}" \
  --top-k 10
