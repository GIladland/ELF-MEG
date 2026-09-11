#!/usr/bin/env bash
#SBATCH --job-name=fmri_mini_txt_ovf
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/fmri_minilm1536_test107_diffusion_overfit-%j.out
#SBATCH --error=submit-jobs/fmri_minilm1536_test107_diffusion_overfit-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
source_dir=$DATA_ROOT/minilm/oracle1536_delayed_seg10_middle10content_clockfix
source_run=minilm_delayed1536_seg10_middle10content_clockfix_meaninit_freezeelf_adapter_train11725_val266_ep30_b32_20260828
source_eval=eval_${source_run}_mri2sem_pred1536

# The MRI2SEM brain decoder is fixed. These are its 107 held-out fMRI
# predictions; only the downstream semantic-to-text bridge is optimized.
export NPZ_PATH=$source_dir/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_pred1536_test_middle10content_clockfix.npz
export SOURCE_NPZ=$NPZ_PATH
export RUN_TAG=${RUN_TAG:-fmri_minilm1536_to_elf_TEST107_TRANSDUCTIVE_DIFFUSION_OVERFIT_ep300_b16_20260829}
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$DATA_ROOT/minilm/eval_latents/${source_eval}_t5small_latents_fp16.npy
export INIT_E2E_CHECKPOINT=${INIT_E2E_CHECKPOINT:-/data/engs-pnpl/glandau/elf-runs/$source_run/best.pt}
export INIT_E2E_ADAPTER_MISMATCH=error

export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export TORCH_PIP_SPEC=${TORCH_PIP_SPEC:-torch==2.7.1}
export TORCH_PIP_INDEX_URL=${TORCH_PIP_INDEX_URL:-https://download.pytorch.org/whl/cu118}

export NUM_EXAMPLES=107
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=107
export RETRIEVAL_NUM_EXAMPLES=107
export RETRIEVAL_EVAL_EVERY=${RETRIEVAL_EVAL_EVERY:-200}
export RETRIEVAL_BATCH_SIZE=256
export EVAL_EVERY=${EVAL_EVERY:-100}
export EPOCHS=${EPOCHS:-300}
export BATCH_SIZE=${BATCH_SIZE:-16}
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_ADAPTER_KIND=flat
export SEMANTIC_INPUT_PROJECTION_DIM=384
export FREEZE_ELF=0
export LAST_N_BLOCKS=-1
export LR=${LR:-1e-4}
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_PROB=0.5
export DECODER_NOISE_SCALE=2.5
export TRAIN_TARGET_MASK_MODE=full
export NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-32}
export TRAIN_TIMESTEP_STEPS=32
export CHECKPOINT_METRIC=negative_wer
export USE_WANDB=1
export WANDB_RUN_GROUP=fmri_minilm1536_to_text_transductive_diffusion_overfit_20260829
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="TRANSDUCTIVE CONTROL, not held-out end-to-end text generalization. The MRI2SEM brain decoder remains fixed and was trained without these four test stories. ELF plus its semantic adapter are deliberately overfit on the 107 fixed fMRI-derived MiniLM1536 predictions and their ten-word labels; best.pt is selected by deterministic same-row WER."
export SAVE_FINAL_CHECKPOINT=1

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
