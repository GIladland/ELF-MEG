#!/usr/bin/env bash
#SBATCH --job-name=elf_scale_x1
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-1%2
#SBATCH --output=submit-jobs/eval_elf_semantic_scaling_x1_exact-%A_%a.out
#SBATCH --error=submit-jobs/eval_elf_semantic_scaling_x1_exact-%A_%a.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1

task_id=${LADDER_TASK_ID_OVERRIDE:-${SLURM_ARRAY_TASK_ID:-0}}
case "$task_id" in
  0)
    modality=mri_minilm; audit_n=266
    oracle_tag=fmri_exact_minilm384_fresh_elfb_full_memorize_all12098_ep600_seed7_20260901
    export NPZ_PATH=$known_dir/oracle_exact_minilm384_train_val.npz
    export VAL_NUM_EXAMPLES=266
    export TARGET_LATENTS_CACHE=$known_dir/oracle_exact_minilm384_trainval_current10_t5small_latents_fp16.npy
    ;;
  1)
    modality=meg_ada; audit_n=110
    oracle_tag=tang_apples_exact_ada002_fresh_elfb_full_memorize_all2788_ep1000_seed7_20260901
    export NPZ_PATH=$DATA_ROOT/tang_themoth_qc4wyals_ada_v2/interface/qc4wyals_val_exact_ada002.npz
    export VAL_NUM_EXAMPLES=0
    export TARGET_LATENTS_CACHE=$DATA_ROOT/tang_themoth_qc4wyals_ada_v2/interface/qc4wyals_val_exact_ada002_t5small_latents_fp16.npy
    ;;
  *) exit 2 ;;
esac
oracle_tag=${ORACLE_TAG_OVERRIDE:-$oracle_tag}

export RUN_TAG=${RUN_TAG_OVERRIDE:-eval_elf_semantic_scaling_${modality}_x1_exact_val${audit_n}_seed49_20260902}
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/$oracle_tag/best.pt
export INIT_E2E_ADAPTER_MISMATCH=error
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export MODEL=${MODEL_OVERRIDE:-ELF-B}
export CHECKPOINT_PATH=${CHECKPOINT_PATH_OVERRIDE:-embedded-language-flows/ELF-B-owt-torch}
export EVAL_ONLY=1
export EPOCHS=0
export NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=$audit_n
export RETRIEVAL_NUM_EXAMPLES=$audit_n
export BATCH_SIZE=32
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_ADAPTER_KIND=flat
export ADAPTER_DROPOUT=0.0
export FREEZE_ELF=1
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CFG_SCALE=1.0
export SELF_COND_CFG_SCALE=1.0
export GENERATION_T5_RETRIEVAL=1
export CHECKPOINT_METRIC=negative_wer
export SEED=49
export USE_WANDB=1
export WANDB_ENTITY=giladland-university-of-oxford
export WANDB_PROJECT=BrainDiffusion
export WANDB_RUN_GROUP=elf_semantic_scaling_20260902
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_SAMPLE_EXAMPLES=16
export WANDB_NOTES="Matched x1 exact-input baseline for the ELF semantic scaling curve. Known text only; no brain vector is loaded."
export TMP_ENV_ROOT=$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118
export ENV_STAMP=braindiffusion-elf-torch271cu118-py310-v1
export SKIP_ARC_SETUP=1

bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"

python_bin=$TMP_ENV_ROOT/bin/python
bert_root=$DATA_ROOT/python-packages/bert-score-0.3.13
export PYTHONNOUSERSITE=1
export PYTHONPATH=$bert_root${PYTHONPATH:+:$PYTHONPATH}
"$python_bin" "$PROJECT_ROOT/scripts/add_bertscore_to_metrics.py" \
  --metrics-json "$OUTPUT_DIR/best_metrics.json" \
  --output "$OUTPUT_DIR/best_metrics.bertscore.json" \
  --batch-size 16 --device "${BERT_DEVICE:-cuda}" --top-k 10
