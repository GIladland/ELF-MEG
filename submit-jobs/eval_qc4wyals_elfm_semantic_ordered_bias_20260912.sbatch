#!/usr/bin/env bash
#SBATCH --job-name=qc4_ordbias
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --array=0-9%1
#SBATCH --output=submit-jobs/eval_qc4wyals_elfm_semantic_ordered_bias_20260912-%A_%a.out
#SBATCH --error=submit-jobs/eval_qc4wyals_elfm_semantic_ordered_bias_20260912-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
runs_root=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
interface=$DATA_ROOT/tang_themoth_qc4wyals_nested_oof_20260908/interface
oracle=$runs_root/elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906/best.pt
semantic_winner=$runs_root/meg_elfm_ctxdistill_robustinit_exactteacher_outputprojector_ep100_seed49_20260906/best_adapter.pt
content_head=${CONTENT_HEAD_CHECKPOINT:?Set CONTENT_HEAD_CHECKPOINT}
ordered_head=${ORDERED_HEAD_CHECKPOINT:?Set ORDERED_HEAD_CHECKPOINT}

task=${SLURM_ARRAY_TASK_ID:-0}
prior_values=(0.0 0.75)
prior_tags=(0 0p75)
bias_values=(1.0 2.0 4.0 8.0 16.0)
bias_tags=(1p0 2p0 4p0 8p0 16p0)
prior=${prior_values[$((task / 5))]}
prior_tag=${prior_tags[$((task / 5))]}
bias=${bias_values[$((task % 5))]}
bias_tag=${bias_tags[$((task % 5))]}

export NPZ_PATH=$interface/qc4wyals_oof_predicted_train_valtail_ada002.npz
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

export SEMANTIC_LEXICAL_HEAD_CHECKPOINT=$content_head
export SEMANTIC_ORDERED_HEAD_CHECKPOINT=$ordered_head
export BRAIN_LEXICAL_TOPK=1
export BRAIN_LEXICAL_DECODE_BIAS_STRENGTH=0.0
export BRAIN_LEXICAL_PRIOR_SUBTRACTION=0.0
export BRAIN_ORDERED_PRIOR_SUBTRACTION=$prior
export BRAIN_ORDERED_MIN_PRIOR_PROBABILITY=0.0
export BRAIN_ORDERED_MIN_MARGIN=0.0
export BRAIN_ORDERED_DECODE_BIAS_STRENGTH=$bias
export BRAIN_ORDERED_DECODE_MAX_POSITIONS=24

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

export RUN_TAG=meg_elfm_puresem_ordbias_p${prior_tag}_b${bias_tag}_seed49_20260912
export OUTPUT_DIR=$runs_root/$RUN_TAG
export TARGET_LATENTS_CACHE=$interface/flat_context_t5small_latents_fp16.npy
export SAVE_FINAL_CHECKPOINT=0
export SAVE_EVAL_CHECKPOINTS=0
export USE_WANDB=0
export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1
export WANDB_NOTES="Pure-semantic ELF-M ordered-word soft decoder bias, nested-OOF predicted-ADA head, prior=${prior}, bias=${bias}; noise start, no external text source, final val110 only, test26 absent."

cd "$PROJECT_ROOT"
bash submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch
"$TMP_ENV_ROOT/bin/python" scripts/cap_generated_text_words.py \
  --metrics-json "$OUTPUT_DIR/best_metrics.json" \
  --output-json "$OUTPUT_DIR/best_metrics.cap10.json" \
  --max-words 10 \
  --run-tag "$RUN_TAG" \
  --checkpoint "$semantic_winner"
