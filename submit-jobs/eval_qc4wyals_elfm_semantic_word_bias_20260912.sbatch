#!/usr/bin/env bash
#SBATCH --job-name=qc4_wordbias
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --array=0-11%1
#SBATCH --output=submit-jobs/eval_qc4wyals_elfm_semantic_word_bias_20260912-%A_%a.out
#SBATCH --error=submit-jobs/eval_qc4wyals_elfm_semantic_word_bias_20260912-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
runs_root=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
interface=$DATA_ROOT/tang_themoth_qc4wyals_nested_oof_20260908/interface
oracle=$runs_root/elf_archscale_meg_ada_x16_elfm_continue_cos2e5_ep8_wcmean_seed7_20260906/best.pt
semantic_winner=$runs_root/meg_elfm_ctxdistill_robustinit_exactteacher_outputprojector_ep100_seed49_20260906/best_adapter.pt
word_head=${WORD_HEAD_CHECKPOINT:?Set WORD_HEAD_CHECKPOINT to the selected nested-OOF semantic word head}

task=${SLURM_ARRAY_TASK_ID:-0}
topk_values=(1 3 5)
bias_values=(4.0 8.0 16.0 32.0)
bias_tags=(4p0 8p0 16p0 32p0)
topk=${topk_values[$((task / 4))]}
bias=${bias_values[$((task % 4))]}
bias_tag=${bias_tags[$((task % 4))]}

for required in \
  "$interface/qc4wyals_oof_predicted_train_valtail_ada002.npz" \
  "$interface/flat_context_t5small_latents_fp16.npy" \
  "$oracle" \
  "$semantic_winner" \
  "$word_head"; do
  [[ -f "$required" ]] || { echo "Missing required input: $required" >&2; exit 2; }
done

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

export SEMANTIC_LEXICAL_HEAD_CHECKPOINT=$word_head
export BRAIN_LEXICAL_TOPK=$topk
export BRAIN_LEXICAL_TEMPERATURE=0.5
export BRAIN_LEXICAL_PRIOR_SUBTRACTION=0.5
export BRAIN_LEXICAL_DECODE_BIAS_STRENGTH=$bias
export BRAIN_LEXICAL_MAX_PRIOR_PROBABILITY=1.0
export BRAIN_LEXICAL_DECODE_BIAS_ONCE=1
export BRAIN_LEXICAL_DECODE_BIAS_MODE=sequence
export BRAIN_LEXICAL_DECODE_BIAS_MAX_POSITIONS=24

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

export RUN_TAG=meg_elfm_puresem_wordbias_top${topk}_b${bias_tag}_prior0p5_seed49_20260912
export OUTPUT_DIR=$runs_root/$RUN_TAG
export TARGET_LATENTS_CACHE=$interface/flat_context_t5small_latents_fp16.npy
export SAVE_FINAL_CHECKPOINT=0
export SAVE_EVAL_CHECKPOINTS=0
export USE_WANDB=0
export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1
export WANDB_NOTES="Pure-semantic ELF-M soft word-evidence bias: nested-OOF-trained ADA word head, predicted ADA only at val inference, noise start, no generated T5 or D'Ascoli source, test26 absent."

cd "$PROJECT_ROOT"
bash submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch
"$TMP_ENV_ROOT/bin/python" scripts/cap_generated_text_words.py \
  --metrics-json "$OUTPUT_DIR/best_metrics.json" \
  --output-json "$OUTPUT_DIR/best_metrics.cap10.json" \
  --max-words 10 \
  --run-tag "$RUN_TAG" \
  --checkpoint "$semantic_winner"
