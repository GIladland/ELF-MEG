#!/usr/bin/env bash
#SBATCH --job-name=fmri_elf_ord
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --array=0-11%4
#SBATCH --output=submit-jobs/eval_fmri_elf_ordered_content_screen-%A_%a.out
#SBATCH --error=submit-jobs/eval_fmri_elf_ordered_content_screen-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
mri_run=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1
source_checkpoint=${SOURCE_CHECKPOINT_OVERRIDE:-/data/engs-pnpl/glandau/elf-runs/fmri_minilm_elf_xattn_last2_lr1e4_from_recover2_step1000_20260831_lexgate_top5_partitioned_dev0p1eval64_fixrepeat4/final.pt}
lexical_head=/data/engs-pnpl/glandau/elf-runs/fmri_supervised_content_probe_20260831/semantic1536_mlp512_priorresidual.pt
ordered_head=${ORDERED_HEAD_OVERRIDE:-/data/engs-pnpl/glandau/elf-runs/fmri_ordered_word_probe_20260831/semantic_mlp512_pair1p0.pt}

ordered_strength=0
ordered_prior=0
ordered_min_prior=0
ordered_min_margin=0
content_max=16
ordered_max=24
case "$SLURM_ARRAY_TASK_ID" in
  0) label=contentearly16_ord0 ;;
  1) ordered_strength=2; label=contentearly16_ord2_p0 ;;
  2) ordered_strength=5; label=contentearly16_ord5_p0 ;;
  3) ordered_strength=10; label=contentearly16_ord10_p0 ;;
  4) ordered_strength=20; label=contentearly16_ord20_p0 ;;
  5) ordered_strength=40; label=contentearly16_ord40_p0 ;;
  6) ordered_strength=5; ordered_prior=0.25; label=contentearly16_ord5_p0p25 ;;
  7) ordered_strength=10; ordered_prior=0.25; label=contentearly16_ord10_p0p25 ;;
  8) ordered_strength=20; ordered_prior=0.25; label=contentearly16_ord20_p0p25 ;;
  9) ordered_strength=10; ordered_prior=0.5; label=contentearly16_ord10_p0p5 ;;
  10) ordered_strength=20; ordered_prior=0.5; label=contentearly16_ord20_p0p5 ;;
  11) ordered_strength=10; content_max=12; label=contentearly12_ord10_p0 ;;
  12) ordered_strength=80; label=contentearly16_ord80_p0 ;;
  13) ordered_strength=120; label=contentearly16_ord120_p0 ;;
  14) ordered_strength=80; ordered_prior=0.25; label=contentearly16_ord80_p0p25 ;;
  15) ordered_strength=120; ordered_prior=0.25; label=contentearly16_ord120_p0p25 ;;
  16) ordered_strength=40; content_max=8; ordered_max=10; label=contentearly8_ord40_p0 ;;
  17) ordered_strength=80; content_max=8; ordered_max=10; label=contentearly8_ord80_p0 ;;
  18) ordered_strength=120; content_max=8; ordered_max=10; label=contentearly8_ord120_p0 ;;
  19) ordered_strength=80; ordered_prior=0.25; content_max=8; ordered_max=10; label=contentearly8_ord80_p0p25 ;;
  20) ordered_strength=40; content_max=10; ordered_max=12; label=contentearly10_ord40_p0 ;;
  21) ordered_strength=80; content_max=10; ordered_max=12; label=contentearly10_ord80_p0 ;;
  22) ordered_strength=80; ordered_min_prior=0.0025; label=contentearly16_ord80_common0p0025 ;;
  23) ordered_strength=80; ordered_min_prior=0.005; label=contentearly16_ord80_common0p005 ;;
  24) ordered_strength=80; ordered_min_prior=0.01; label=contentearly16_ord80_common0p01 ;;
  25) ordered_strength=80; ordered_min_prior=0.02; label=contentearly16_ord80_common0p02 ;;
  26) ordered_strength=80; ordered_min_prior=0.005; content_max=8; ordered_max=10; label=contentearly8_ord80_common0p005 ;;
  27) ordered_strength=80; ordered_min_prior=0.01; content_max=8; ordered_max=10; label=contentearly8_ord80_common0p01 ;;
  28) ordered_strength=80; ordered_min_margin=0.5; label=contentearly16_ord80_margin0p5 ;;
  29) ordered_strength=80; ordered_min_margin=1.0; label=contentearly16_ord80_margin1 ;;
  30) ordered_strength=80; ordered_min_margin=2.0; label=contentearly16_ord80_margin2 ;;
  31) ordered_strength=80; ordered_min_margin=3.0; label=contentearly16_ord80_margin3 ;;
  32) ordered_strength=80; ordered_min_margin=1.0; content_max=8; ordered_max=10; label=contentearly8_ord80_margin1 ;;
  33) ordered_strength=80; ordered_min_margin=2.0; content_max=8; ordered_max=10; label=contentearly8_ord80_margin2 ;;
  34) ordered_strength=80; ordered_min_margin=3.0; content_max=8; ordered_max=10; label=contentearly8_ord80_margin3 ;;
  *) echo "Unsupported task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

eval_label=${EVAL_LABEL_OVERRIDE:-eval64}
export RUN_TAG=${RUN_PREFIX_OVERRIDE:-fmri_minilm_elf_ordered_content}_${label}_${eval_label}_20260901
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export NPZ_PATH=$known_dir/oracle_exact_minilm384_train_val.npz
export SOURCE_NPZ=$NPZ_PATH
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export BRAIN_INPUT_NPZ=$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz
export BRAIN_TRAIN_KEY=train_x
export BRAIN_VAL_KEY=val_x
export BRAIN_VAL_ROLL=${BRAIN_VAL_ROLL_OVERRIDE:-0}
export BRAIN_MODEL_CHECKPOINT=$source_checkpoint
export BRAIN_SEMANTIC_OUTPUT_DIM=1536
export BRAIN_PROJECTOR_MISMATCH=error
export BRAIN_HIDDEN_DIM=2048
export BRAIN_RES_BLOCKS=4
export BRAIN_DROPOUT=0.1
export BRAIN_UNFREEZE=frozen
export BRAIN_LR=0

export BRAIN_LEXICAL_HEAD_CHECKPOINT=$lexical_head
export BRAIN_LEXICAL_TOPK=${CONTENT_TOPK_OVERRIDE:-10}
export BRAIN_LEXICAL_TEMPERATURE=${CONTENT_TEMPERATURE_OVERRIDE:-2.0}
export BRAIN_LEXICAL_CONTEXT_MODE=partitioned
export BRAIN_LEXICAL_PRIOR_SUBTRACTION=${CONTENT_PRIOR_OVERRIDE:-0.1}
export BRAIN_LEXICAL_DECODE_BIAS_STRENGTH=${CONTENT_BIAS_OVERRIDE:-40}
export BRAIN_LEXICAL_MAX_PRIOR_PROBABILITY=1.0
export BRAIN_LEXICAL_DECODE_BIAS_ONCE=1
export BRAIN_LEXICAL_DECODE_BIAS_MODE=${CONTENT_MODE_OVERRIDE:-scattered}
export BRAIN_LEXICAL_DECODE_BIAS_MAX_POSITIONS=$content_max
export BRAIN_ORDERED_HEAD_CHECKPOINT=$ordered_head
export BRAIN_ORDERED_PRIOR_SUBTRACTION=$ordered_prior
export BRAIN_ORDERED_MIN_PRIOR_PROBABILITY=$ordered_min_prior
export BRAIN_ORDERED_MIN_MARGIN=$ordered_min_margin
export BRAIN_ORDERED_DECODE_BIAS_STRENGTH=$ordered_strength
export BRAIN_ORDERED_DECODE_MAX_POSITIONS=$ordered_max

export SEMANTIC_ADAPTER_KIND=residual_delay_tokens
export SEMANTIC_MAPPER_HIDDEN_DIM=1024
export SEMANTIC_DELAY_TOKENS=4
export SEMANTIC_DELAY_LAYERS=2
export SEMANTIC_DELAY_HEADS=8
export ADAPTER_DROPOUT=0.1
export SEMANTIC_INPUT_PROJECTION_DIM=0
export FREEZE_SEMANTIC_ADAPTER=1
export INIT_E2E_CHECKPOINT=$source_checkpoint
export INIT_E2E_ADAPTER_MISMATCH=error

export FREEZE_ELF=1
export ELF_LORA_RANK=4
export ELF_LORA_ALPHA=8
export ELF_LORA_DROPOUT=0.0
export ELF_LORA_TARGETS=attention,mlp
export ELF_LORA_LAST_N_BLOCKS=2
export ELF_BRAIN_CROSS_ATTENTION_LAST_N_BLOCKS=2
export ELF_BRAIN_CROSS_ATTENTION_HEADS=12
export ELF_BRAIN_CROSS_ATTENTION_DROPOUT=0.0
export ELF_BRAIN_CROSS_ATTENTION_ONLY=1
export ELF_BRAIN_CROSS_ATTENTION_DECODER_ONLY=1

export TARGET_LATENTS_CACHE=$known_dir/oracle_exact_minilm384_trainval_current10_t5small_latents_fp16.npy
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export SKIP_ARC_SETUP=1
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=266
export EVAL_NUM_EXAMPLES=${EVAL_NUM_EXAMPLES_OVERRIDE:-64}
export RETRIEVAL_NUM_EXAMPLES=-1
export GENERATION_T5_RETRIEVAL=0
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
export WANDB_NOTES="Leakage-safe val64 ordered-prior/content-first ELF diffusion screen. Ordered and content heads were fit on train11725 only. No candidate reranking; test107 never loaded."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
