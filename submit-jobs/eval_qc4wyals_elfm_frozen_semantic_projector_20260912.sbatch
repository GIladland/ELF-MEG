#!/usr/bin/env bash
#SBATCH --job-name=qc4_semproj
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --output=submit-jobs/eval_qc4wyals_elfm_frozen_semantic_projector_20260912-%j.out
#SBATCH --error=submit-jobs/eval_qc4wyals_elfm_frozen_semantic_projector_20260912-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
runs_root=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
interface=$DATA_ROOT/tang_themoth_qc4wyals_nested_oof_20260908/interface
oracle=$runs_root/elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906/best.pt
exact_flat_projector=$runs_root/elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906_flat_adapter_val110_wcmean_seed49/best.pt

for required in \
  "$interface/qc4wyals_oof_predicted_train_valtail_ada002.npz" \
  "$interface/flat_context_t5small_latents_fp16.npy" \
  "$oracle" \
  "$exact_flat_projector"; do
  [[ -f "$required" ]] || { echo "Missing required input: $required" >&2; exit 2; }
done

export NPZ_PATH=$interface/qc4wyals_oof_predicted_train_valtail_ada002.npz
export SOURCE_NPZ=$NPZ_PATH
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export MODEL=ELF-M
export CHECKPOINT_PATH=$oracle
export INIT_E2E_CHECKPOINT=$exact_flat_projector
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
export EVAL_TARGET_LENGTH=22
export EVAL_NUM_EXAMPLES=110
export RETRIEVAL_NUM_EXAMPLES=110
export RETRIEVAL_BATCH_SIZE=64
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CFG_SCALE=1.0
export SELF_COND_CFG_SCALE=1.0
export GENERATION_T5_RETRIEVAL=1
export SEED=49

export RUN_TAG=meg_elfm_puresem_exactflat_projector_seed49_20260912
export OUTPUT_DIR=$runs_root/$RUN_TAG
export TARGET_LATENTS_CACHE=$interface/flat_context_t5small_latents_fp16.npy
export SAVE_FINAL_CHECKPOINT=0
export SAVE_EVAL_CHECKPOINTS=0
export USE_WANDB=0
export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1
export WANDB_NOTES="Frozen predicted qc4wyals ADA into the pre-MEG-tuning exact-ADA flat projector; pure noise-start ELF-M, val110 only, test26 absent."

cd "$PROJECT_ROOT"
bash submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch
"$TMP_ENV_ROOT/bin/python" scripts/cap_generated_text_words.py \
  --metrics-json "$OUTPUT_DIR/best_metrics.json" \
  --output-json "$OUTPUT_DIR/best_metrics.cap10.json" \
  --max-words 10 \
  --run-tag "$RUN_TAG" \
  --checkpoint "$exact_flat_projector"
