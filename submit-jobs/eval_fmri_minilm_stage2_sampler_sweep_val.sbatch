#!/usr/bin/env bash
#SBATCH --job-name=fmri_s2_sample
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --array=0-11%4
#SBATCH --output=submit-jobs/eval_fmri_minilm_stage2_sampler_sweep_val-%A_%a.out
#SBATCH --error=submit-jobs/eval_fmri_minilm_stage2_sampler_sweep_val-%A_%a.err

set -euo pipefail

# Frozen val266 sampler sweep. The test107 brain array is never read.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
mri_run=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1
baseline_run=fmri_minilm_rawbrain_e2e_projector_long100_resdelaytok_h1024_l2_h8_storybal_b32_seed49_20260830
baseline_checkpoint=/data/engs-pnpl/glandau/elf-runs/$baseline_run/best.pt
eval_checkpoint=${EVAL_CHECKPOINT_OVERRIDE:-$baseline_checkpoint}
sampler_label=${SAMPLER_LABEL:-baseline}

seeds=(7 23 49 101)
cfgs=(0.5 1.0 1.5)
seed_index=$((SLURM_ARRAY_TASK_ID % 4))
cfg_index=$((SLURM_ARRAY_TASK_ID / 4))
seed=${seeds[$seed_index]}
cfg=${cfgs[$cfg_index]}
cfg_tag=${cfg/./p}
sampling_steps=${NUM_SAMPLING_STEPS_OVERRIDE:-32}

export RUN_TAG=fmri_minilm_stage2_sampler_${sampler_label}_seed${seed}_cfg${cfg_tag}_steps${sampling_steps}_val266_20260830
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export NPZ_PATH=$known_dir/oracle_exact_minilm384_train_val.npz
export SOURCE_NPZ=$NPZ_PATH
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export TRAIN_BALANCE_KEY=story

export BRAIN_INPUT_NPZ=$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz
export BRAIN_TRAIN_KEY=train_x
export BRAIN_VAL_KEY=val_x
export BRAIN_MODEL_CHECKPOINT=$eval_checkpoint
export BRAIN_SEMANTIC_OUTPUT_DIM=1536
export BRAIN_HIDDEN_DIM=2048
export BRAIN_RES_BLOCKS=4
export BRAIN_DROPOUT=0.1
export BRAIN_UNFREEZE=frozen

export SEMANTIC_ADAPTER_KIND=residual_delay_tokens
export SEMANTIC_MAPPER_HIDDEN_DIM=1024
export SEMANTIC_DELAY_TOKENS=4
export SEMANTIC_DELAY_LAYERS=2
export SEMANTIC_DELAY_HEADS=8
export ADAPTER_DROPOUT=0.1
export SEMANTIC_INPUT_PROJECTION_DIM=0
export INIT_E2E_CHECKPOINT=$eval_checkpoint
export INIT_E2E_ADAPTER_MISMATCH=error
export SEMANTIC_ALIGNMENT_TARGET_NPZ=$known_dir/oracle_exact_minilm384_train_val.npz
export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings

export TARGET_LATENTS_CACHE=$known_dir/oracle_exact_minilm384_trainval_current10_t5small_latents_fp16.npy
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export SKIP_ARC_SETUP=1
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=266
export EVAL_NUM_EXAMPLES=266
export RETRIEVAL_NUM_EXAMPLES=266
export RETRIEVAL_BATCH_SIZE=256
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export FREEZE_ELF=1
export EVAL_ONLY=1
export CFG_SCALE=$cfg
export SELF_COND_CFG_SCALE=1.0
export NUM_SAMPLING_STEPS=$sampling_steps
export TRAIN_TIMESTEP_STEPS=32
export SEED=$seed
export USE_WANDB=0
export SAVE_FINAL_CHECKPOINT=0
export SAVE_EVAL_CHECKPOINTS=0
export WANDB_NOTES="Frozen stage-2 validation sampler sweep: checkpoint=$eval_checkpoint label=$sampler_label seed=$seed cfg=$cfg steps=32. val266 only; test107 remains closed."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
