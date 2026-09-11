#!/usr/bin/env bash
#SBATCH --job-name=elf_scale_raw
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-5%4
#SBATCH --output=submit-jobs/eval_elf_semantic_scaling_raw_ladder-%A_%a.out
#SBATCH --error=submit-jobs/eval_elf_semantic_scaling_raw_ladder-%A_%a.err

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

export RUN_TAG=${RUN_TAG_OVERRIDE:-eval_${oracle_tag}_rawbrain_seed49}
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export INIT_E2E_CHECKPOINT=$oracle
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export MODEL=${MODEL_OVERRIDE:-ELF-B}
export CHECKPOINT_PATH=${CHECKPOINT_PATH_OVERRIDE:-embedded-language-flows/ELF-B-owt-torch}
export EVAL_ONLY=1
export EPOCHS=0
export NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=$audit_n
export RETRIEVAL_NUM_EXAMPLES=$audit_n
export BATCH_SIZE=32
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export ADAPTER_DROPOUT=0.0
export FREEZE_ELF=1
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CFG_SCALE=1.0
export SELF_COND_CFG_SCALE=1.0
export GENERATION_T5_RETRIEVAL=1
export CHECKPOINT_METRIC=content_word_overlap
export SEED=49

if [[ "$modality" == mri_minilm ]]; then
  export NPZ_PATH=$DATA_ROOT/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz
  export SOURCE_NPZ=$NPZ_PATH
  export SEMANTIC_ALIGNMENT_TARGET_NPZ=$known_dir/oofmix_exact_minilm384_targets_train23450_val266.npz
  export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings
  export VAL_NUM_EXAMPLES=266
  export TARGET_LATENTS_CACHE=$known_dir/oofmix_current10_t5small_latents_fp16.npy
  export SEMANTIC_ADAPTER_KIND=residual_mlp
  export SEMANTIC_MAPPER_HIDDEN_DIM=512
  export SEMANTIC_DELAY_TOKENS=4
  export INIT_E2E_ADAPTER_MISMATCH=residual-mlp
else
  interface=$DATA_ROOT/tang_themoth_qc4wyals_ada_v2/interface
  export NPZ_PATH=$interface/qc4wyals_val_predicted_ada002.npz
  export SOURCE_NPZ=$NPZ_PATH
  export SEMANTIC_ALIGNMENT_TARGET_NPZ=$interface/qc4wyals_val_exact_ada002.npz
  export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings
  export VAL_NUM_EXAMPLES=0
  export TARGET_LATENTS_CACHE=$interface/qc4wyals_val_predicted_ada002_t5small_latents_fp16.npy
  export SEMANTIC_ADAPTER_KIND=flat
  export INIT_E2E_ADAPTER_MISMATCH=error
fi

export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=elf_semantic_scaling_20260902
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_SAMPLE_EXAMPLES=16
export WANDB_NOTES="Fixed raw-brain rung for semantic-corpus scaling. Frozen scaled ELF and untrained identity-preserving input route; validation brain vectors are evaluation-only. MRI test107 and MEG test26 are not loaded."
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
