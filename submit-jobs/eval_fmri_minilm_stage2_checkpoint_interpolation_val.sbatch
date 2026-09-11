#!/usr/bin/env bash
#SBATCH --job-name=fmri_s2_interp
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-4%5
#SBATCH --output=submit-jobs/eval_fmri_minilm_stage2_checkpoint_interpolation_val-%A_%a.out
#SBATCH --error=submit-jobs/eval_fmri_minilm_stage2_checkpoint_interpolation_val-%A_%a.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
RUNS_ROOT=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
ENV_ROOT=${ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
baseline_run=fmri_minilm_rawbrain_e2e_projector_long100_resdelaytok_h1024_l2_h8_storybal_b32_seed49_20260830
candidate_run=fmri_minilm_stage2_inputonly_frozenelf_contentw3_bow1p0_midlr_dec2_from_delaytok_best_ep100_seed49_20260830
baseline_checkpoint=${INTERP_BASELINE_CHECKPOINT:-$RUNS_ROOT/$baseline_run/best.pt}
candidate_checkpoint=${INTERP_CANDIDATE_CHECKPOINT:-$RUNS_ROOT/$candidate_run/best.pt}
interp_label=${INTERP_LABEL:-base_to_content}

case "$SLURM_ARRAY_TASK_ID" in
  0) alpha=0.10; alpha_tag=0p10 ;;
  1) alpha=0.25; alpha_tag=0p25 ;;
  2) alpha=0.50; alpha_tag=0p50 ;;
  3) alpha=0.75; alpha_tag=0p75 ;;
  4) alpha=0.90; alpha_tag=0p90 ;;
  *) echo "Unsupported task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

tmp_root=${SLURM_TMPDIR:-/tmp}
interpolated_checkpoint=$tmp_root/fmri_stage2_interp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.pt
if ! command -v conda >/dev/null 2>&1; then
  source /apps/system/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_ROOT"
python "$PROJECT_ROOT/scripts/interpolate_e2e_checkpoints.py" \
  --baseline "$baseline_checkpoint" \
  --candidate "$candidate_checkpoint" \
  --alpha "$alpha" \
  --output "$interpolated_checkpoint"

known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
mri_run=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1
export RUN_TAG=fmri_minilm_stage2_interp_${interp_label}_alpha${alpha_tag}_seed49_val266_20260830
export OUTPUT_DIR=$RUNS_ROOT/$RUN_TAG
export NPZ_PATH=$known_dir/oracle_exact_minilm384_train_val.npz
export SOURCE_NPZ=$NPZ_PATH
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export BRAIN_INPUT_NPZ=$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz
export BRAIN_TRAIN_KEY=train_x
export BRAIN_VAL_KEY=val_x
export BRAIN_MODEL_CHECKPOINT=$baseline_checkpoint
export BRAIN_SEMANTIC_OUTPUT_DIM=1536
export BRAIN_HIDDEN_DIM=2048
export BRAIN_RES_BLOCKS=4
export BRAIN_DROPOUT=0.1
export BRAIN_UNFREEZE=frozen
export BRAIN_LR=0
export SEMANTIC_ADAPTER_KIND=residual_delay_tokens
export SEMANTIC_MAPPER_HIDDEN_DIM=1024
export SEMANTIC_DELAY_TOKENS=4
export SEMANTIC_DELAY_LAYERS=2
export SEMANTIC_DELAY_HEADS=8
export ADAPTER_DROPOUT=0.1
export SEMANTIC_INPUT_PROJECTION_DIM=0
export NORMALIZE_SEMANTIC_PROJECTION=0
export INIT_E2E_CHECKPOINT=$interpolated_checkpoint
export INIT_E2E_ADAPTER_MISMATCH=error
export SEMANTIC_ALIGNMENT_TARGET_NPZ=$known_dir/oracle_exact_minilm384_train_val.npz
export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings
export TARGET_LATENTS_CACHE=$known_dir/oracle_exact_minilm384_trainval_current10_t5small_latents_fp16.npy
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$ENV_ROOT}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export SKIP_ARC_SETUP=1
export FREEZE_ELF=1
export EVAL_ONLY=1
export USE_WANDB=0
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=266
export EVAL_NUM_EXAMPLES=266
export RETRIEVAL_NUM_EXAMPLES=266
export RETRIEVAL_BATCH_SIZE=256
export BATCH_SIZE=32
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export EPOCHS=1
export LR=1e-6
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CFG_SCALE=1.0
export SEED=49
export SAVE_FINAL_CHECKPOINT=0
export WANDB_NOTES="Validation-only checkpoint interpolation alpha=$alpha; no reference text used to construct checkpoint; test107 closed."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
