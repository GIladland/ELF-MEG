#!/usr/bin/env bash
#SBATCH --job-name=eval_tgpt3072
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/eval_native3072_tang_gpt_oracle_vs_mri2sem-%j.out
#SBATCH --error=submit-jobs/eval_native3072_tang_gpt_oracle_vs_mri2sem-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
export RUN_TAG=${RUN_TAG:-native3072_tang_gpt_oracle_seg10_freezeelf_30ep_b16_20260824}
export TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/$RUN_TAG}
export CHECKPOINT=${CHECKPOINT:-$TRAIN_OUTPUT_DIR/best.pt}
export ORACLE_TEST_NPZ=${ORACLE_TEST_NPZ:-$DATA_ROOT/tang_gpt/oracle3072/uts01_tang_gpt_lanczos_seg10_oracle3072_test.npz}
export PREDICTED_TEST_NPZ=${PREDICTED_TEST_NPZ:-$DATA_ROOT/mri2sem/uts01_lanczos_seg10_test_pred3072.npz}
export EVAL_ROOT=${EVAL_ROOT:-$TRAIN_OUTPUT_DIR/test_oracle_vs_mri2sem}
export EVAL_LATENTS_DIR=${EVAL_LATENTS_DIR:-$DATA_ROOT/tang_gpt/oracle3072}

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing native 3072 checkpoint: $CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$ORACLE_TEST_NPZ" || ! -f "$PREDICTED_TEST_NPZ" ]]; then
  echo "Missing paired oracle or predicted test NPZ." >&2
  exit 2
fi

run_eval() {
  local name=$1
  local npz_path=$2
  local output_dir=$3
  local latent_cache=$4

  SOURCE_NPZ="$npz_path" \
  NPZ_PATH="$npz_path" \
  OUTPUT_DIR="$output_dir" \
  TARGET_LATENTS_CACHE="$latent_cache" \
  INIT_E2E_CHECKPOINT="$CHECKPOINT" \
  INIT_E2E_ADAPTER_MISMATCH=error \
  FREEZE_ELF=1 \
  EVAL_ONLY=1 \
  USE_WANDB=0 \
  SAVE_FINAL_CHECKPOINT=0 \
  EPOCHS=1 \
  BATCH_SIZE=16 \
  VAL_NUM_EXAMPLES=0 \
  EVAL_NUM_EXAMPLES=0 \
  RETRIEVAL_NUM_EXAMPLES=0 \
  RETRIEVAL_EVAL_EVERY=1 \
  RETRIEVAL_BATCH_SIZE=256 \
  CONTEXT_LENGTH=64 \
  SEMANTIC_HIDDEN_DIM=4096 \
  DENOISER_LOSS_WEIGHT=0.1 \
  DECODER_LOSS_WEIGHT=5.0 \
  DECODER_PROB=0.9 \
  DECODER_NOISE_SCALE=1.0 \
  CFG_SCALE=1.0 \
  SELF_COND_CFG_SCALE=1.0 \
  SEED=7 \
  WANDB_RUN_NAME="${RUN_TAG}_${name}_test" \
  WANDB_NOTES="Paired native-3072 ELF test evaluation: ${name}." \
  bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
}

mkdir -p "$EVAL_ROOT"
run_eval \
  oracle \
  "$ORACLE_TEST_NPZ" \
  "$EVAL_ROOT/oracle" \
  "$EVAL_LATENTS_DIR/${RUN_TAG}_oracle_test_t5small_latents_fp16.npy"
run_eval \
  mri2sem_predicted \
  "$PREDICTED_TEST_NPZ" \
  "$EVAL_ROOT/mri2sem_predicted" \
  "$EVAL_LATENTS_DIR/${RUN_TAG}_predicted_test_t5small_latents_fp16.npy"

"$TMP_ENV_ROOT/bin/python" "$PROJECT_ROOT/scripts/compare_native3072_elf_evals.py" \
  --oracle-dir "$EVAL_ROOT/oracle" \
  --predicted-dir "$EVAL_ROOT/mri2sem_predicted" \
  --output "$EVAL_ROOT/oracle_vs_mri2sem_comparison.json"
