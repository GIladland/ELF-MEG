#!/usr/bin/env bash
#SBATCH --job-name=elfm_interp
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --array=0-4%3
#SBATCH --output=submit-jobs/eval_qc4wyals_elfm_interpolation_ladder_20260906-%A_%a.out
#SBATCH --error=submit-jobs/eval_qc4wyals_elfm_interpolation_ladder_20260906-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
interface=$data_root/tang_themoth_qc4wyals_ada_v2/interface
ladder=$interface/interpolation_val110_20260906
oracle=/data/engs-pnpl/glandau/elf-runs/elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906/best.pt
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
alphas=(0 0p25 0p5 0p75 1)
alpha=${alphas[${SLURM_ARRAY_TASK_ID:-0}]}

[[ -f "$ladder/interpolation_ladder.json" ]] || {
  echo "Interpolation packs have not been prepared in $ladder" >&2
  exit 2
}

export RUN_TAG=eval_qc4wyals_val110_elfm_cos_exactinterp_alpha${alpha}_seed49_20260906
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export NPZ_PATH=$ladder/qc4wyals_val_interp_alpha${alpha}_ada002.npz
export SOURCE_NPZ=$NPZ_PATH
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export MODEL=ELF-M
export CHECKPOINT_PATH=embedded-language-flows/ELF-M-owt-torch
export INIT_E2E_CHECKPOINT=$oracle
export INIT_E2E_ADAPTER_MISMATCH=error
export SEMANTIC_ADAPTER_KIND=flat
export SEMANTIC_HIDDEN_DIM=4096
export CONTEXT_LENGTH=64
export FREEZE_ELF=1
export FREEZE_SEMANTIC_ADAPTER=1
export EVAL_ONLY=1
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=110
export RETRIEVAL_NUM_EXAMPLES=110
export GENERATION_T5_RETRIEVAL=1
export BATCH_SIZE=22
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export SAVE_FINAL_CHECKPOINT=0
export SAVE_EVAL_CHECKPOINTS=0
export SEED=49
export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=qc4wyals_elfm_interpolation_ladder_20260906
export WANDB_RUN_NAME=$RUN_TAG
export TMP_ENV_ROOT=$data_root/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1
export TARGET_LATENTS_CACHE=$ladder/val110_alpha${alpha}_t5small_latents_fp16.npy
export WANDB_NOTES="Evaluation-only predicted-to-exact ADA interpolation on the same held-out val110 candidates. alpha=0 is raw qc4wyals and alpha=1 is exact ADA; protected test26 is absent."

exec bash "$project_root/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
