#!/usr/bin/env bash
#SBATCH --job-name=qc4_ctxproj
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-2%1
#SBATCH --output=submit-jobs/train_qc4wyals_elfm_oof_context_projector_20260912-%A_%a.out
#SBATCH --error=submit-jobs/train_qc4wyals_elfm_oof_context_projector_20260912-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
runs_root=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
interface=$DATA_ROOT/tang_themoth_qc4wyals_nested_oof_20260908/interface
oracle=$runs_root/elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906/best.pt
exact_flat=$runs_root/elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906_flat_adapter_val110_wcmean_seed49/best.pt
meg_tuned=$runs_root/meg_elfm_ctxdistill_robustinit_exactteacher_outputprojector_ep100_seed49_20260906/best_adapter.pt

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    arm=meginit_lr1e6
    init_checkpoint=$meg_tuned
    adapter_lr=1e-6
    ;;
  1)
    arm=meginit_lr3e6
    init_checkpoint=$meg_tuned
    adapter_lr=3e-6
    ;;
  2)
    arm=exactinit_lr1e6
    init_checkpoint=$exact_flat
    adapter_lr=1e-6
    ;;
  *) exit 2 ;;
esac

for required in \
  "$interface/qc4wyals_oof_predicted_train_valtail_ada002.npz" \
  "$interface/qc4wyals_oof_exact_alignment_train_valtail_ada002.npz" \
  "$interface/flat_context_t5small_latents_fp16.npy" \
  "$oracle" "$init_checkpoint"; do
  [[ -f "$required" ]] || { echo "Missing required input: $required" >&2; exit 2; }
done

export NPZ_PATH=$interface/qc4wyals_oof_predicted_train_valtail_ada002.npz
export SOURCE_NPZ=$NPZ_PATH
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export SEMANTIC_ALIGNMENT_TARGET_NPZ=$interface/qc4wyals_oof_exact_alignment_train_valtail_ada002.npz
export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings
export MODEL=ELF-M
export CHECKPOINT_PATH=$oracle
export INIT_E2E_CHECKPOINT=$init_checkpoint
export INIT_E2E_ADAPTER_MISMATCH=error
export SEMANTIC_ADAPTER_KIND=flat
export SEMANTIC_HIDDEN_DIM=4096
export CONTEXT_LENGTH=64
export ADAPTER_DROPOUT=0.0
export FREEZE_SEMANTIC_ADAPTER=0
export FREEZE_ELF=1
export ELF_LORA_RANK=0
export ELF_BRAIN_CROSS_ATTENTION_LAST_N_BLOCKS=0

export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=110
export EVAL_TARGET_LENGTH=22
export EVAL_NUM_EXAMPLES=110
export RETRIEVAL_NUM_EXAMPLES=-1
export RETRIEVAL_EVAL_EVERY=0
export RETRIEVAL_BATCH_SIZE=64
export BATCH_SIZE=16
export EPOCHS=12
export EVAL_EVERY=250
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export TRAIN_TARGET_MASK_MODE=full
export COND_DROPOUT_PROB=0.0
export DENOISER_LOSS_WEIGHT=1.0
export DECODER_LOSS_WEIGHT=1.0
export DECODER_PROB=0.5
export DECODER_NOISE_SCALE=2.5
export DECODER_CONTENT_TOKEN_WEIGHT=1.5
export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.02
export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.0
export DECODER_CONDITION_PAIRING_MARGIN_WEIGHT=0.01
export DECODER_CONDITION_PAIRING_MARGIN=0.10
export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=0.0
export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.0
export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.10
export LR=$adapter_lr
export LR_SCHEDULE=cosine
export WARMUP_STEPS=100
export MIN_LR_RATIO=0.05
export CHECKPOINT_METRIC=word_content_mean
export CFG_SCALE=1.0
export SELF_COND_CFG_SCALE=1.0
export GENERATION_T5_RETRIEVAL=1
export SEED=49

export RUN_TAG=meg_elfm_puresem_oofctx_${arm}_ep12_seed49_20260912
export OUTPUT_DIR=$runs_root/$RUN_TAG
export TARGET_LATENTS_CACHE=$interface/flat_context_t5small_latents_fp16.npy
export SAVE_FINAL_CHECKPOINT=0
export SAVE_EVAL_CHECKPOINTS=0
export EVAL_CHECKPOINT_TOP_K=1
export USE_WANDB=0
export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1
export WANDB_NOTES="Pure-semantic nested-OOF predicted-ADA context-projector adaptation (${arm}); frozen qc4wyals inputs and frozen ELF-M, noise-start val110, test26 absent."

cd "$PROJECT_ROOT"
exec bash submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch
