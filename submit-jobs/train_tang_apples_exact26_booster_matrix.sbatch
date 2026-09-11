#!/usr/bin/env bash
#SBATCH --job-name=ta_exact26
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-5%4
#SBATCH --output=submit-jobs/train_tang_apples_exact26_booster_matrix-%A_%a.out
#SBATCH --error=submit-jobs/train_tang_apples_exact26_booster_matrix-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
interface_dir=$DATA_ROOT/tang_themoth_qc4wyals_ada_v2/interface
runs_root=/data/engs-pnpl/glandau/elf-runs
content_run=$runs_root/tang_apples_ada002_elfb_full_content_knowntext_all2788_seed49_20260901
standard_run=$runs_root/tang_apples_ada002_elfb_full_standard_knowntext_all2788_seed49_20260901

export INIT_E2E_CHECKPOINT=$content_run/best.pt
if [[ "${BOOSTER_FROM_BASE:-0}" == "1" ]]; then
  export INIT_E2E_CHECKPOINT=
elif [[ -n "${BOOSTER_INIT_CHECKPOINT:-}" ]]; then
  export INIT_E2E_CHECKPOINT=$BOOSTER_INIT_CHECKPOINT
fi
export FREEZE_ELF=1
export LAST_N_BLOCKS=-1
export ELF_LORA_RANK=0
export ELF_LORA_ALPHA=16
export ELF_LORA_DROPOUT=0.05
export ELF_LORA_TARGETS=attention,mlp
export ELF_LORA_LAST_N_BLOCKS=-1
export LR=2e-5
export ELF_LR=0

case "$SLURM_ARRAY_TASK_ID" in
  0)
    arm=contentbest_adapter
    ;;
  1)
    arm=contentbest_lora8
    export ELF_LORA_RANK=8
    export ELF_LR=5e-6
    ;;
  2)
    arm=contentbest_last2
    export FREEZE_ELF=0
    export LAST_N_BLOCKS=2
    export ELF_LR=3e-6
    ;;
  3)
    arm=contentbest_full
    export FREEZE_ELF=0
    export LAST_N_BLOCKS=-1
    export LR=1e-5
    export ELF_LR=1e-6
    ;;
  4)
    arm=contentstep4000_full
    export INIT_E2E_CHECKPOINT=$content_run/checkpoints/eval_step_00004000_score_0.193537.pt
    export FREEZE_ELF=0
    export LAST_N_BLOCKS=-1
    export LR=1e-5
    export ELF_LR=1e-6
    ;;
  5)
    arm=standardstep2000_full
    export INIT_E2E_CHECKPOINT=$standard_run/checkpoints/eval_step_00002000_score_0.188403.pt
    export FREEZE_ELF=0
    export LAST_N_BLOCKS=-1
    export LR=1e-5
    export ELF_LR=1e-6
    ;;
  *) exit 2 ;;
esac

booster_epochs=${BOOSTER_EPOCHS:-1500}
booster_suffix=${BOOSTER_SUFFIX:-}
if [[ -n "$booster_suffix" ]]; then
  booster_suffix=_$booster_suffix
fi
export RUN_TAG=tang_apples_exact26_booster_${arm}_ep${booster_epochs}${booster_suffix}_seed49_20260901
export OUTPUT_DIR=$runs_root/$RUN_TAG
export SOURCE_NPZ=$interface_dir/qc4wyals_test_exact_ada002.npz
export NPZ_PATH=$SOURCE_NPZ
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export MODEL=ELF-B
export CHECKPOINT_PATH=embedded-language-flows/ELF-B-owt-torch
export INIT_E2E_ADAPTER_MISMATCH=error
export SEMANTIC_ADAPTER_KIND=flat
export SEMANTIC_INPUT_PROJECTION_DIM=0
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export ADAPTER_DROPOUT=0.0
export NORMALIZE_SEMANTIC_PROJECTION=0
export FREEZE_SEMANTIC_ADAPTER=0
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=26
export RETRIEVAL_NUM_EXAMPLES=26
export GENERATION_T5_RETRIEVAL=1
export EPOCHS=$booster_epochs
export BATCH_SIZE=16
export EVAL_EVERY=${BOOSTER_EVAL_EVERY:-250}
export RETRIEVAL_EVAL_EVERY=${BOOSTER_EVAL_EVERY:-250}
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export LR_SCHEDULE=cosine
export WARMUP_STEPS=100
export MIN_LR_RATIO=0.01
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_PROB=0.5
export DECODER_NOISE_SCALE=2.5
export DECODER_CONTENT_TOKEN_WEIGHT=2.0
export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.10
export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.05
export TRAIN_TARGET_MASK_MODE=full
export CHECKPOINT_METRIC=content_word_overlap
export SAVE_FINAL_CHECKPOINT=1
export SAVE_EVAL_CHECKPOINTS=1
export EVAL_CHECKPOINT_TOP_K=3
export EVAL_CHECKPOINT_EVERY=1
export SEED=49
export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=tang_apples_exact26_booster_20260901
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_SAMPLE_EXAMPLES=26
export TARGET_LATENTS_CACHE=$interface_dir/qc4wyals_test26_t5small_latents_fp16.npy
export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1
export WANDB_NOTES="Intentional exact-ADA/text booster on all 26 BirthOfANation targets after full-corpus Tang/Apples pretraining. Protected qc4wyals MEG-predicted test vectors are absent from this job."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
