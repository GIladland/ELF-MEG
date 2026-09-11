#!/usr/bin/env bash
#SBATCH --job-name=fmri_pareto5
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/rerank_fmri_stage2_pareto5_semantic_val-%j.out
#SBATCH --error=submit-jobs/rerank_fmri_stage2_pareto5_semantic_val-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
RUNS_ROOT=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
ENV_ROOT=${ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
BERTSCORE_PACKAGE_ROOT=${BERTSCORE_PACKAGE_ROOT:-$DATA_ROOT/python-packages/bert-score-0.3.13}
output_dir=$RUNS_ROOT/fmri_stage2_pareto5_semantic_rerank_val266_20260830
baseline=$RUNS_ROOT/fmri_minilm_rawbrain_e2e_projector_long100_resdelaytok_h1024_l2_h8_storybal_b32_seed49_20260830
lora=$RUNS_ROOT/fmri_minilm_stage2_recover_lora_last2_r4_content_from_delaytok_best_ep100_seed49_20260830_recovery_from_contentwinner
recovery=$RUNS_ROOT/fmri_minilm_stage2_recover2_lora_last2_r4_semantic_strong_from_delaytok_best_ep100_seed49_20260830_from_lora_step3000
source_npz=$RUNS_ROOT/fmri_minilm_stage2_sampler_seed7_cfg0p5_steps32_val266_20260830/eval_semantic_predictions.npz

export HF_HOME=${HF_HOME:-$DATA_ROOT/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export PYTHONNOUSERSITE=1
export PYTHONPATH=$BERTSCORE_PACKAGE_ROOT${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

if ! command -v conda >/dev/null 2>&1; then
  source /apps/system/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_ROOT"
mkdir -p "$output_dir"
cd "$PROJECT_ROOT"

python scripts/rerank_generated_candidates_by_semantic.py \
  --source-npz "$source_npz" \
  --metrics "$baseline/eval_step_035000.json" \
  --metrics "$lora/eval_step_002000.json" \
  --metrics "$lora/eval_step_003000.json" \
  --metrics "$recovery/eval_step_000250.json" \
  --metrics "$recovery/eval_step_000500.json" \
  --output "$output_dir/best_metrics.json" \
  --device cpu \
  --local-files-only

python scripts/add_bertscore_to_metrics.py \
  --metrics-json "$output_dir/best_metrics.json" \
  --output "$output_dir/best_metrics.bertscore.json" \
  --batch-size 8 \
  --device cpu \
  --top-k 10
