#!/usr/bin/env bash
#SBATCH --job-name=fmri_roberta_map
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=submit-jobs/rerank_fmri_stage2_roberta_mapper_val-%j.out
#SBATCH --error=submit-jobs/rerank_fmri_stage2_roberta_mapper_val-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
RUNS_ROOT=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
ENV_ROOT=${ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
BERTSCORE_PACKAGE_ROOT=${BERTSCORE_PACKAGE_ROOT:-$DATA_ROOT/python-packages/bert-score-0.3.13}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
output_dir=$RUNS_ROOT/fmri_stage2_pareto5_roberta_mapper_rerank_val266_20260830
baseline=$RUNS_ROOT/fmri_minilm_rawbrain_e2e_projector_long100_resdelaytok_h1024_l2_h8_storybal_b32_seed49_20260830
lora=$RUNS_ROOT/fmri_minilm_stage2_recover_lora_last2_r4_content_from_delaytok_best_ep100_seed49_20260830_recovery_from_contentwinner
recovery=$RUNS_ROOT/fmri_minilm_stage2_recover2_lora_last2_r4_semantic_strong_from_delaytok_best_ep100_seed49_20260830_from_lora_step3000
val_semantic=$RUNS_ROOT/fmri_minilm_stage2_sampler_seed7_cfg0p5_steps32_val266_20260830/eval_semantic_predictions.npz

export HF_HOME=${HF_HOME:-$DATA_ROOT/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PROJECT_ROOT/scripts:$BERTSCORE_PACKAGE_ROOT${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
if ! command -v conda >/dev/null 2>&1; then
  source /apps/system/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_ROOT"
mkdir -p "$output_dir"
cd "$PROJECT_ROOT"

python scripts/rerank_candidates_by_roberta_mapper.py \
  --train-npz "$known_dir/oracle_exact_minilm384_train_val.npz" \
  --val-semantic-npz "$val_semantic" \
  --metrics "$baseline/eval_step_035000.json" \
  --metrics "$lora/eval_step_002000.json" \
  --metrics "$lora/eval_step_003000.json" \
  --metrics "$recovery/eval_step_000250.json" \
  --metrics "$recovery/eval_step_001000.json" \
  --cache "$known_dir/train11725_roberta_large_layer17_mean_fp32.npy" \
  --output "$output_dir/best_metrics.json" \
  --length-penalty 0.001 \
  --length-penalty 0.002 \
  --length-penalty 0.005 \
  --length-penalty 0.01 \
  --length-penalty 0.02 \
  --length-penalty 0.05 \
  --device cuda \
  --local-files-only

python scripts/add_bertscore_to_metrics.py \
  --metrics-json "$output_dir/best_metrics.json" \
  --output "$output_dir/best_metrics.bertscore.json" \
  --batch-size 16 \
  --device cuda \
  --top-k 10
