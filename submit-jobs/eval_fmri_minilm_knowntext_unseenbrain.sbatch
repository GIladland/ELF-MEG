#!/usr/bin/env bash
#SBATCH --job-name=eval_fmri_known
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --array=0-5
#SBATCH --output=submit-jobs/eval_fmri_minilm_knowntext_unseenbrain-%A_%a.out
#SBATCH --error=submit-jobs/eval_fmri_minilm_knowntext_unseenbrain-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
source_dir=$DATA_ROOT/minilm/oracle1536_delayed_seg10_middle10content_clockfix
oracle_run=fmri_minilm_knowntext_exact384_all12098_testfirst_full_k64_ep100_b64_seed49_20260829
adapter_run=fmri_minilm_knowntext_oofmix1536_adapteronly_ep10_b32_seed49_20260829
e2e_run=${E2E_RUN:-fmri_minilm_knowntext_oofmix1536_full_e2e_ep10_b32_seed49_20260829}
linear_mapper=fmri_minilm_knowntext_oofmix1536_to_exact384_linear_train23450_val266_20260829
mlp_mapper=fmri_minilm_knowntext_oofmix1536_to_exact384_mlp1024_train23450_val266_20260829

case "$SLURM_ARRAY_TASK_ID" in
  0)
    condition=known_oracle_exact384
    export NPZ_PATH=$known_dir/oracle_exact_minilm384_test107.npz
    init_run=$oracle_run
    projection_dim=0
    ;;
  1)
    condition=unseenbrain_mapped_linear384
    export NPZ_PATH=/data/engs-pnpl/glandau/elf-runs/$linear_mapper/mapped_mri2sem_test_minilm384.npz
    init_run=$oracle_run
    projection_dim=0
    ;;
  2)
    condition=unseenbrain_mapped_mlp384
    export NPZ_PATH=/data/engs-pnpl/glandau/elf-runs/$mlp_mapper/mapped_mri2sem_test_minilm384.npz
    init_run=$oracle_run
    projection_dim=0
    ;;
  3)
    condition=unseenbrain_direct1536_adapteronly
    export NPZ_PATH=$source_dir/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_pred1536_test_middle10content_clockfix.npz
    init_run=$adapter_run
    projection_dim=384
    ;;
  4)
    condition=unseenbrain_direct1536_full_e2e
    export NPZ_PATH=$source_dir/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_pred1536_test_middle10content_clockfix.npz
    init_run=$e2e_run
    projection_dim=384
    ;;
  5)
    condition=oracle_delayed1536_full_e2e_control
    export NPZ_PATH=$source_dir/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_oracle1536_middle10content_clockfix_test.npz
    init_run=$e2e_run
    projection_dim=384
    ;;
  *) exit 2 ;;
esac

export SOURCE_NPZ=$NPZ_PATH
export RUN_TAG=eval_${init_run}_${condition}_test107_20260829
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$known_dir/eval_latents/${RUN_TAG}_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/$init_run/best.pt
export INIT_E2E_ADAPTER_MISMATCH=error

export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export NUM_EXAMPLES=107
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=107
export RETRIEVAL_NUM_EXAMPLES=107
export EVAL_EVERY=1
export RETRIEVAL_EVAL_EVERY=1
export RETRIEVAL_BATCH_SIZE=256
export EPOCHS=1
export BATCH_SIZE=16
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_ADAPTER_KIND=flat
export SEMANTIC_INPUT_PROJECTION_DIM=$projection_dim
export FREEZE_ELF=1
export EVAL_ONLY=1
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CHECKPOINT_METRIC=negative_wer
export SEED=49
export USE_WANDB=1
export WANDB_RUN_GROUP=fmri_minilm_knowntext_unseenbrain_final_test107_20260829
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Final 107-row $condition evaluation. The target sentences were included in the perfect-MiniLM known-text stage, but the 107 MRI2SEM-predicted vectors were excluded from mapper, adapter, E2E training, validation, and checkpoint selection."
export SAVE_FINAL_CHECKPOINT=0

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
