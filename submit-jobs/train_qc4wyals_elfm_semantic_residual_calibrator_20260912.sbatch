#!/usr/bin/env bash
#SBATCH --job-name=qc4_semcal
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-1%1
#SBATCH --output=submit-jobs/train_qc4wyals_elfm_semantic_residual_calibrator_20260912-%A_%a.out
#SBATCH --error=submit-jobs/train_qc4wyals_elfm_semantic_residual_calibrator_20260912-%A_%a.err

set -euo pipefail

# Pure predicted-ADA calibration. The zero-output residual mapper makes step
# zero identical to the current semantic-only ELF-M winner. Its parent context
# net and all ELF weights remain frozen; inference continues to start at noise.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
interface=$DATA_ROOT/tang_themoth_qc4wyals_nested_oof_20260908/interface
runs_root=/data/engs-pnpl/glandau/elf-runs
oracle=$runs_root/elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906/best.pt
semantic_winner=$runs_root/meg_elfm_ctxdistill_robustinit_exactteacher_outputprojector_ep100_seed49_20260906/best_adapter.pt

export SOURCE_NPZ=$interface/qc4wyals_oof_predicted_train_valtail_ada002.npz
export NPZ_PATH=$SOURCE_NPZ
export SEMANTIC_ALIGNMENT_TARGET_NPZ=$interface/qc4wyals_oof_exact_alignment_train_valtail_ada002.npz
export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export MODEL=ELF-M
export CHECKPOINT_PATH=$oracle
export INIT_E2E_CHECKPOINT=$semantic_winner
export INIT_E2E_ADAPTER_MISMATCH=direct-identity
export SEMANTIC_ADAPTER_KIND=residual_identity
export SEMANTIC_MAPPER_HIDDEN_DIM=512
export SEMANTIC_HIDDEN_DIM=4096
export CONTEXT_LENGTH=64
export ADAPTER_DROPOUT=0.05
export TRAIN_SEMANTIC_INPUT_ONLY=1
export FREEZE_ELF=1
export ELF_LORA_RANK=0

export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=110
export EVAL_NUM_EXAMPLES=110
export RETRIEVAL_NUM_EXAMPLES=110
export GENERATION_T5_RETRIEVAL=1
export BATCH_SIZE=64
export EPOCHS=40
export EVAL_EVERY=250
export RETRIEVAL_EVAL_EVERY=250
export RETRIEVAL_BATCH_SIZE=64
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CFG_SCALE=1.0
export SELF_COND_CFG_SCALE=1.0

export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=1.0
export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=0.05
export SEMANTIC_CONTEXT_LOSS_WEIGHT=0.25
export SEMANTIC_RAW_ANCHOR_LOSS_WEIGHT=0.10
export SEMANTIC_GEOMETRY_LOSS_WEIGHT=0.02
export SEMANTIC_RANK_DISTILL_LOSS_WEIGHT=0.02
export LR=1e-4
export WEIGHT_DECAY=0.001
export LR_SCHEDULE=cosine
export WARMUP_STEPS=100
export MIN_LR_RATIO=0.05

task=${SLURM_ARRAY_TASK_ID:-0}
case "$task" in
  0)
    arm=semobj
    export SEMANTIC_OBJECTIVE_ONLY=1
    export DENOISER_LOSS_WEIGHT=0
    export DECODER_LOSS_WEIGHT=0
    ;;
  1)
    arm=flowjoint
    export SEMANTIC_OBJECTIVE_ONLY=0
    export BATCH_SIZE=32
    export EPOCHS=20
    export LR=3e-5
    export DENOISER_LOSS_WEIGHT=1.0
    export DECODER_LOSS_WEIGHT=0.25
    export DECODER_PROB=0.5
    export DECODER_P_MEAN=-1.5
    export DECODER_P_STD=0.8
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_CONDITION_PAIRING_MARGIN_WEIGHT=0.05
    export DECODER_CONDITION_PAIRING_MARGIN=0.10
    ;;
  *)
    echo "Unsupported task $task" >&2
    exit 2
    ;;
esac

for required in "$SOURCE_NPZ" "$SEMANTIC_ALIGNMENT_TARGET_NPZ" "$oracle" "$semantic_winner"; do
  [[ -f "$required" ]] || { echo "Missing required input: $required" >&2; exit 2; }
done

export RUN_TAG=meg_elfm_puresem_nestedoof_residcal_${arm}_ep${EPOCHS}_seed49_20260912
export OUTPUT_DIR=$runs_root/$RUN_TAG
export TARGET_LATENTS_CACHE=$interface/flat_context_t5small_latents_fp16.npy
export CHECKPOINT_METRIC=word_content_mean
export SAVE_TRAINABLE_ONLY_CHECKPOINT=0
export SAVE_FINAL_CHECKPOINT=0
export SAVE_EVAL_CHECKPOINTS=0
export USE_WANDB=0
export SEED=49
export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1
export WANDB_NOTES="Nested-OOF residual calibration of predicted qc4wyals ADA for frozen noise-start ELF-M. No D'Ascoli or generated T5 source; val110 only; test26 absent."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
