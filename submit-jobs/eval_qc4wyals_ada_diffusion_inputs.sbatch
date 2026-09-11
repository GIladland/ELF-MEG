#!/usr/bin/env bash
#SBATCH --job-name=qc4w_eval
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-9%4
#SBATCH --output=submit-jobs/eval_qc4wyals_ada_diffusion_inputs-%A_%a.out
#SBATCH --error=submit-jobs/eval_qc4wyals_ada_diffusion_inputs-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
interface_dir=$DATA_ROOT/tang_themoth_qc4wyals_ada_v2/interface
calibration_dir=$DATA_ROOT/tang_themoth_qc4wyals_ada_v2/calibration
mapper_root=/data/engs-pnpl/glandau/elf-runs
eval_split=${EVAL_SPLIT:-test}

case "$eval_split" in
  test)
    split_rows=26
    split_source_prefix=qc4wyals_test
    ;;
  val|validation)
    eval_split=val
    split_rows=110
    split_source_prefix=qc4wyals_val
    ;;
  *)
    echo "EVAL_SPLIT must be test or val, got: $eval_split" >&2
    exit 2
    ;;
esac

: "${SOURCE_CHECKPOINT:?Set SOURCE_CHECKPOINT to the selected ADA-to-ELF checkpoint}"
if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
  echo "Missing SOURCE_CHECKPOINT=$SOURCE_CHECKPOINT" >&2
  exit 2
fi

case "$SLURM_ARRAY_TASK_ID" in
  0)
    input_name=exact_ada
    source_npz=$interface_dir/${split_source_prefix}_exact_ada002.npz
    ;;
  1)
    input_name=raw_qc4wyals
    source_npz=$interface_dir/${split_source_prefix}_predicted_ada002.npz
    ;;
  2)
    input_name=linear
    mapper_dir=$mapper_root/qc4wyals_ada_mapper_linear_quick120_seed49_20260901
    [[ "$eval_split" == test ]] && source_npz=$mapper_dir/mapped_oracle_test_semantic1536.npz || source_npz=$mapper_dir/mapped_val_semantic1536.npz
    ;;
  3)
    input_name=mlp512
    mapper_dir=$mapper_root/qc4wyals_ada_mapper_mlp512_quick120_seed49_20260901
    [[ "$eval_split" == test ]] && source_npz=$mapper_dir/mapped_oracle_test_semantic1536.npz || source_npz=$mapper_dir/mapped_val_semantic1536.npz
    ;;
  4)
    input_name=reslinear
    mapper_dir=$mapper_root/qc4wyals_ada_mapper_reslinear_quick120_seed49_20260901
    [[ "$eval_split" == test ]] && source_npz=$mapper_dir/mapped_oracle_test_semantic1536.npz || source_npz=$mapper_dir/mapped_val_semantic1536.npz
    ;;
  5)
    input_name=resmlp512
    mapper_dir=$mapper_root/qc4wyals_ada_mapper_resmlp512_quick120_seed49_20260901
    [[ "$eval_split" == test ]] && source_npz=$mapper_dir/mapped_oracle_test_semantic1536.npz || source_npz=$mapper_dir/mapped_val_semantic1536.npz
    ;;
  6)
    input_name=knn64_ndcg
    source_npz=$calibration_dir/qc4wyals_calibrated_ndcg_${eval_split}_ada002.npz
    ;;
  7)
    input_name=diagonal_collapse_control
    source_npz=$calibration_dir/qc4wyals_calibrated_cosine_${eval_split}_ada002.npz
    ;;
  8)
    input_name=mean_shift
    source_npz=$calibration_dir/qc4wyals_calibrated_mean_shift_${eval_split}_ada002.npz
    ;;
  9)
    input_name=diagonal_balanced
    source_npz=$calibration_dir/qc4wyals_calibrated_diagonal_balanced_${eval_split}_ada002.npz
    ;;
  *) exit 2 ;;
esac

checkpoint_tag=$(basename "$SOURCE_CHECKPOINT" .pt)
checkpoint_parent=$(basename "$(dirname "$SOURCE_CHECKPOINT")")
eval_suffix=${EVAL_TAG_SUFFIX:-}
if [[ -n "$eval_suffix" ]]; then
  eval_suffix=_$eval_suffix
fi
export RUN_TAG=eval_qc4wyals_${eval_split}${split_rows}_${input_name}_through_${checkpoint_parent}_${checkpoint_tag}${eval_suffix}_seed49_20260901
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export SOURCE_NPZ=$source_npz
export NPZ_PATH=$source_npz
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export MODEL=ELF-B
export CHECKPOINT_PATH=embedded-language-flows/ELF-B-owt-torch
export INIT_E2E_CHECKPOINT=$SOURCE_CHECKPOINT
export INIT_E2E_ADAPTER_MISMATCH=error
export SEMANTIC_ADAPTER_KIND=flat
export SEMANTIC_INPUT_PROJECTION_DIM=0
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=${SOURCE_SEMANTIC_HIDDEN_DIM:-4096}
export ADAPTER_DROPOUT=0.0
export FREEZE_ELF=1
export FREEZE_SEMANTIC_ADAPTER=1
export EVAL_ONLY=1
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=$split_rows
export RETRIEVAL_NUM_EXAMPLES=$split_rows
export GENERATION_T5_RETRIEVAL=1
export EPOCHS=0
export BATCH_SIZE=${EVAL_BATCH_SIZE:-22}
export EVAL_EVERY=1
export RETRIEVAL_EVAL_EVERY=1
export NUM_SAMPLING_STEPS=${EVAL_SAMPLING_STEPS:-32}
export TRAIN_TIMESTEP_STEPS=32
export CFG_SCALE=${EVAL_CFG_SCALE:-1.0}
export SELF_COND_CFG_SCALE=1.0
export CHECKPOINT_METRIC=content_word_overlap
export SAVE_FINAL_CHECKPOINT=0
export SAVE_EVAL_CHECKPOINTS=0
export SEED=49
export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=qc4wyals_ada_diffusion_input_eval_20260901
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_SAMPLE_EXAMPLES=${WANDB_SAMPLE_EXAMPLES:-20}
# Use one cache per source pack so concurrently launched exact/raw evaluations
# cannot race while atomically replacing the same cache sidecar.
export TARGET_LATENTS_CACHE=$interface_dir/qc4wyals_${eval_split}${split_rows}_${input_name}_t5small_latents_fp16.npy
export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1
export WANDB_NOTES="Frozen single-checkpoint diffusion comparison on the ${eval_split} split (${split_rows} rows). Exact ADA and raw qc4wyals use the identical checkpoint and decode settings. No fitting occurs in this job."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
