#!/usr/bin/env bash
#SBATCH --job-name=fmri_s2_ensemble
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=submit-jobs/rerank_fmri_minilm_stage2_checkpoint_ensemble_val-%j.out
#SBATCH --error=submit-jobs/rerank_fmri_minilm_stage2_checkpoint_ensemble_val-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
RUNS_ROOT=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
ENV_ROOT=${ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
PYTHON_BIN=${PYTHON_BIN:-$ENV_ROOT/bin/python}
BERTSCORE_PACKAGE_ROOT=${BERTSCORE_PACKAGE_ROOT:-$DATA_ROOT/python-packages/bert-score-0.3.13}
OUTPUT_DIR=$RUNS_ROOT/fmri_minilm_stage2_checkpoint_ensemble_semantic_rerank_val266_20260830

export HF_HOME=${HF_HOME:-$DATA_ROOT/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export PYTHONNOUSERSITE=1
export PYTHONPATH=$BERTSCORE_PACKAGE_ROOT${PYTHONPATH:+:$PYTHONPATH}

mkdir -p "$OUTPUT_DIR" "$BERTSCORE_PACKAGE_ROOT" "$DATA_ROOT/pip-cache"
if [[ ! -f "$BERTSCORE_PACKAGE_ROOT/bert_score/__init__.py" ]]; then
  PIP_CACHE_DIR=$DATA_ROOT/pip-cache "$PYTHON_BIN" -m pip install \
    --no-user --no-deps --target "$BERTSCORE_PACKAGE_ROOT" bert-score==0.3.13
fi

source_run=$RUNS_ROOT/fmri_minilm_stage2_sampler_seed7_cfg0p5_steps32_val266_20260830
source_npz=$source_run/eval_semantic_predictions.npz
if [[ ! -f "$source_npz" ]]; then
  echo "Missing model-predicted semantic archive: $source_npz" >&2
  exit 1
fi

baseline_run=$RUNS_ROOT/fmri_minilm_rawbrain_e2e_projector_long100_resdelaytok_h1024_l2_h8_storybal_b32_seed49_20260830
candidate_files=()
while IFS= read -r path; do
  candidate_files+=("$path")
done < <(
  find \
    "$baseline_run" \
    "$RUNS_ROOT"/fmri_minilm_stage2_full_elf_from_delaytok_best_ep100_seed49_20260830 \
    "$RUNS_ROOT"/fmri_minilm_stage2_lora_attn_r8_from_delaytok_best_ep100_seed49_20260830 \
    "$RUNS_ROOT"/fmri_minilm_stage2_lora_attn_mlp_r8_from_delaytok_best_ep100_seed49_20260830 \
    "$RUNS_ROOT"/fmri_minilm_stage2_contentw3_dec2_lora_attn_r8_from_delaytok_best_ep100_seed49_20260830 \
    "$RUNS_ROOT"/fmri_minilm_stage2_contentw3_dec2_lora_attn_mlp_r8_from_delaytok_best_ep100_seed49_20260830 \
    "$RUNS_ROOT"/fmri_minilm_stage2_inputonly_frozenelf_contentw2_bow1p0_stdlr_from_delaytok_best_ep100_seed49_20260830 \
    "$RUNS_ROOT"/fmri_minilm_stage2_inputonly_frozenelf_contentw2_bow1p0_lowlr_from_delaytok_best_ep100_seed49_20260830 \
    "$RUNS_ROOT"/fmri_minilm_stage2_inputonly_frozenelf_contentw2_bow1p0_lowlr_from_delaytok_best_ep100_seed49_20260830_frombowstep1000 \
    "$RUNS_ROOT"/fmri_minilm_stage2_inputonly_frozenelf_contentw2_bow2p0_lowlr_from_delaytok_best_ep100_seed49_20260830 \
    "$RUNS_ROOT"/fmri_minilm_stage2_inputonly_frozenelf_contentw2_bow1p0_ultralowlr_from_delaytok_best_ep100_seed49_20260830 \
    "$RUNS_ROOT"/fmri_minilm_stage2_inputonly_frozenelf_contentw2_bow1p0_midlr_from_delaytok_best_ep100_seed49_20260830 \
    "$RUNS_ROOT"/fmri_minilm_stage2_inputonly_frozenelf_contentw2_bow0p5_lowlr_from_delaytok_best_ep100_seed49_20260830 \
    "$RUNS_ROOT"/fmri_minilm_stage2_inputonly_frozenelf_contentw3_bow1p0_midlr_dec2_from_delaytok_best_ep100_seed49_20260830 \
    "$RUNS_ROOT"/fmri_minilm_stage2_inputonly_frozenelf_contentw2_bow1p0_midlr_nosemaux_from_delaytok_best_ep100_seed49_20260830 \
    "$RUNS_ROOT"/fmri_minilm_stage2_fullchain_fulladapter_frozenelf_from_delaytok_best_ep150_seed49_20260830 \
    "$RUNS_ROOT"/fmri_minilm_stage2_fullchain_fulladapter_lora_attnmlp_r8_from_delaytok_best_ep150_seed49_20260830 \
    "$RUNS_ROOT"/fmri_minilm_stage2_fullchain_fulladapter_lora_attnmlp_r16_contentw2_dec1p5_from_delaytok_best_ep150_seed49_20260830 \
    "$RUNS_ROOT"/fmri_minilm_stage2_fullchain_fulladapter_frozenelf_contentw2_dec1p5_from_delaytok_best_ep150_seed49_20260830 \
    "$RUNS_ROOT"/fmri_minilm_stage2_fullchain_fulladapter_frozenelf_contentw2_bow0p1_dec1p5_from_delaytok_best_ep150_seed49_20260830 \
    "$RUNS_ROOT"/fmri_minilm_stage2_fullchain_fulladapter_lora_attnmlp_r16_contentw2_bow0p1_dec1p5_from_delaytok_best_ep150_seed49_20260830 \
    "$RUNS_ROOT"/fmri_minilm_stage2_fullchain_fulladapter_frozenelf_contentw2_bow1p0_dec1p5_from_delaytok_best_ep150_seed49_20260830 \
    "$RUNS_ROOT"/fmri_minilm_stage2_fullchain_fulladapter_lora_attnmlp_r16_contentw2_bow1p0_dec1p5_from_delaytok_best_ep150_seed49_20260830 \
    -maxdepth 1 -type f -name 'eval_step_*.json' -print | sort
)
for cfg in 0p5 1p0 1p5; do
  for seed in 7 23 49 101; do
    candidate_files+=("$RUNS_ROOT/fmri_minilm_stage2_sampler_seed${seed}_cfg${cfg}_steps32_val266_20260830/best_metrics.json")
  done
done
if (( ${#candidate_files[@]} < 12 )); then
  echo "Unexpectedly small candidate pool: ${#candidate_files[@]}" >&2
  exit 1
fi

metrics_args=()
for metrics in "${candidate_files[@]}"; do
  if [[ ! -f "$metrics" ]]; then
    echo "Missing candidate metrics: $metrics" >&2
    exit 1
  fi
  metrics_args+=(--metrics "$metrics")
done

cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/rerank_generated_candidates_by_semantic.py \
  --source-npz "$source_npz" \
  "${metrics_args[@]}" \
  --output "$OUTPUT_DIR/best_metrics.json" \
  --device cuda \
  --local-files-only

"$PYTHON_BIN" scripts/add_bertscore_to_metrics.py \
  --metrics-json "$OUTPUT_DIR/best_metrics.json" \
  --output "$OUTPUT_DIR/best_metrics.bertscore.json" \
  --batch-size 16 \
  --device cuda \
  --top-k 10

printf '%s\n' "${candidate_files[@]}" > "$OUTPUT_DIR/candidate_files.txt"
echo "Checkpoint-ensemble semantic reranking written to $OUTPUT_DIR"
