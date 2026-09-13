#!/usr/bin/env bash
#SBATCH --job-name=qc4_elfm_dec
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --array=0-8%1
#SBATCH --output=submit-jobs/eval_qc4wyals_elfm_puresem_decode_sweep_20260912-%A_%a.out
#SBATCH --error=submit-jobs/eval_qc4wyals_elfm_puresem_decode_sweep_20260912-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
interface=$DATA_ROOT/tang_themoth_qc4wyals_ada_v2/interface
runs_root=/data/engs-pnpl/glandau/elf-runs
oracle=$runs_root/elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906/best.pt
semantic_winner=$runs_root/meg_elfm_ctxdistill_robustinit_exactteacher_outputprojector_ep100_seed49_20260906/best_adapter.pt

task=${SLURM_ARRAY_TASK_ID:-0}
cfg_index=$((task / 3))
step_index=$((task % 3))
cfgs=(0.5 1.0 1.5)
cfg_tags=(0p5 1p0 1p5)
steps=(16 32 64)
cfg=${cfgs[$cfg_index]}
cfg_tag=${cfg_tags[$cfg_index]}
sampling_steps=${steps[$step_index]}

for required in \
  "$interface/qc4wyals_predicted_train_valtail_ada002.npz" \
  "$interface/qc4wyals_predicted_train_valtail_t5small_latents_fp16.npy" \
  "$oracle" \
  "$semantic_winner"; do
  [[ -f "$required" ]] || { echo "Missing required input: $required" >&2; exit 2; }
done

export NPZ_PATH=$interface/qc4wyals_predicted_train_valtail_ada002.npz
export SOURCE_NPZ=$NPZ_PATH
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export MODEL=ELF-M
export CHECKPOINT_PATH=$oracle
export INIT_E2E_CHECKPOINT=$semantic_winner
export INIT_E2E_ADAPTER_MISMATCH=error
export SEMANTIC_ADAPTER_KIND=flat
export SEMANTIC_HIDDEN_DIM=4096
export CONTEXT_LENGTH=64
export ADAPTER_DROPOUT=0.0
export FREEZE_SEMANTIC_ADAPTER=1
export FREEZE_ELF=1
export ELF_LORA_RANK=0
export ELF_BRAIN_CROSS_ATTENTION_LAST_N_BLOCKS=0

export EVAL_ONLY=1
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=110
export EVAL_NUM_EXAMPLES=110
export RETRIEVAL_NUM_EXAMPLES=110
export RETRIEVAL_BATCH_SIZE=64
export NUM_SAMPLING_STEPS=$sampling_steps
export TRAIN_TIMESTEP_STEPS=$sampling_steps
export CFG_SCALE=$cfg
export SELF_COND_CFG_SCALE=1.0
export GENERATION_T5_RETRIEVAL=1
export SEED=49

export RUN_TAG=meg_elfm_puresem_decode_ctx27_s${sampling_steps}_cfg${cfg_tag}_seed49_20260912
export OUTPUT_DIR=$runs_root/$RUN_TAG
export TARGET_LATENTS_CACHE=$interface/qc4wyals_predicted_train_valtail_t5small_latents_fp16.npy
export SAVE_FINAL_CHECKPOINT=0
export SAVE_EVAL_CHECKPOINTS=0
export USE_WANDB=0
export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1
export WANDB_NOTES="Frozen pure-semantic ELF-M decode sweep on the exact 27-token combined-pack contract: qc4wyals predicted ADA only, noise start, steps=${sampling_steps}, CFG=${cfg}; final val110 only; no D'Ascoli or generated T5 source; test26 absent."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
