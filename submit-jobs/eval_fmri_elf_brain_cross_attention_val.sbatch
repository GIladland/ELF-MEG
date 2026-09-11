#!/usr/bin/env bash
#SBATCH --job-name=fmri_xattn_ctl
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --array=0-5%3
#SBATCH --output=submit-jobs/eval_fmri_elf_brain_cross_attention_val-%A_%a.out
#SBATCH --error=submit-jobs/eval_fmri_elf_brain_cross_attention_val-%A_%a.err

set -euo pipefail

# One matched val266 generation plus five fixed held-out-brain derangements.
# The checkpoint and adapter layout must be fixed before this control is run.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
mri_run=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1

eval_checkpoint=${EVAL_CHECKPOINT_OVERRIDE:?Set EVAL_CHECKPOINT_OVERRIDE to the selected cross-attention checkpoint}
eval_label=${EVAL_LABEL_OVERRIDE:?Set EVAL_LABEL_OVERRIDE to a stable checkpoint label}
cross_blocks=${CROSS_BLOCKS_OVERRIDE:?Set CROSS_BLOCKS_OVERRIDE to the checkpoint architecture}

rolls=(0 1 7 31 89 137)
roll=${rolls[$SLURM_ARRAY_TASK_ID]}
if [[ "$roll" == "0" ]]; then
  control=matched
else
  control=roll${roll}
fi

export RUN_TAG=eval_fmri_elf_xattn_${eval_label}_${control}_val266_seed49_20260831
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export NPZ_PATH=$known_dir/oracle_exact_minilm384_train_val.npz
export SOURCE_NPZ=$NPZ_PATH
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence

export BRAIN_INPUT_NPZ=$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz
export BRAIN_TRAIN_KEY=train_x
export BRAIN_VAL_KEY=val_x
export BRAIN_VAL_ROLL=$roll
export BRAIN_MODEL_CHECKPOINT=$eval_checkpoint
export BRAIN_SEMANTIC_OUTPUT_DIM=1536
export BRAIN_PROJECTOR_MISMATCH=error
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
export FREEZE_SEMANTIC_ADAPTER=1
export INIT_E2E_CHECKPOINT=$eval_checkpoint
export INIT_E2E_ADAPTER_MISMATCH=error

export FREEZE_ELF=1
export ELF_LORA_RANK=4
export ELF_LORA_ALPHA=8
export ELF_LORA_DROPOUT=0.0
export ELF_LORA_TARGETS=attention,mlp
export ELF_LORA_LAST_N_BLOCKS=2
export ELF_BRAIN_CROSS_ATTENTION_LAST_N_BLOCKS=$cross_blocks
export ELF_BRAIN_CROSS_ATTENTION_HEADS=12
export ELF_BRAIN_CROSS_ATTENTION_DROPOUT=0.0
export ELF_BRAIN_CROSS_ATTENTION_ONLY=1
export ELF_BRAIN_CROSS_ATTENTION_DECODER_ONLY=${CROSS_DECODER_ONLY_OVERRIDE:-0}

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
export EVAL_ONLY=1
export CFG_SCALE=1.0
export SELF_COND_CFG_SCALE=1.0
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export SEED=49
export USE_WANDB=0
export SAVE_FINAL_CHECKPOINT=0
export SAVE_EVAL_CHECKPOINTS=0
export WANDB_NOTES="Frozen ELF cross-attention val266 control checkpoint=$eval_checkpoint control=$control. A nonzero roll deranges only held-out brain rows after alignment verification. test107 is never loaded."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
