#!/usr/bin/env bash
#SBATCH --job-name=meg_t5cgrid1
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-131%8
#SBATCH --output=submit-jobs/sweep_meg_t5_confidence_semantic_grid_20260913-%A_%a.out
#SBATCH --error=submit-jobs/sweep_meg_t5_confidence_semantic_grid_20260913-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
runs_root=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
interface=$data_root/tang_themoth_qc4wyals_nested_oof_20260908/interface
source_npz=$interface/qc4wyals_oof_predicted_train_valtail_ada002.npz
t5_source=$runs_root/meg_t5_source_confidence_val110_20260912/beam4_min10_max20_lp0p8.json
selected_checkpoint=$runs_root/qc4wyals_elfb_dascoli_lora8_mid_ep40_seed49_20260911/best.pt

# The earlier partial-flow search fixed the threshold at 0.20.  This grid
# measures the missing interaction between proposal confidence and semantic
# strength at the successful flow endpoint.  Values are global validation
# hyperparameters; no row-specific target information enters generation.
copy_thresholds=(0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24 0.26 0.28 0.30 0.35)
semantic_scales=(0.0 0.02 0.035 0.05 0.075 0.10 0.15 0.20 0.25 0.35 1.0)
flow_end_time=0.25

task_id=${SLURM_ARRAY_TASK_ID:-0}
semantic_count=${#semantic_scales[@]}
copy_threshold=${copy_thresholds[$((task_id / semantic_count))]}
semantic_scale=${semantic_scales[$((task_id % semantic_count))]}
run_tag=qc4wyals_elfb_t5hybrid_grid1_threshold${copy_threshold}_sem${semantic_scale}_flowend${flow_end_time}_seed49_20260913
output_dir=$runs_root/$run_tag

for required in "$python_bin" "$source_npz" "$t5_source" "$selected_checkpoint"; do
  [[ -f "$required" ]] || { echo "Missing required path: $required" >&2; exit 2; }
done
mkdir -p "$output_dir" "$data_root/tmp"
export HF_HOME=$data_root/huggingface TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=$data_root/tmp

cd "$project_root"
exec "$python_bin" -u scripts/train_npz_semantic_to_elf.py \
  --npz-path "$source_npz" \
  --output_dir "$output_dir" \
  --input-key input_embeddings \
  --sentence-key sentence \
  --model ELF-B \
  --checkpoint_path embedded-language-flows/ELF-B-owt-torch \
  --init-e2e-checkpoint "$selected_checkpoint" \
  --semantic-adapter-kind flat \
  --semantic-hidden-dim 4096 \
  --context_length 64 \
  --freeze-semantic-adapter \
  --elf-lora-rank 8 \
  --elf-lora-alpha 16 \
  --elf-lora-dropout 0.05 \
  --elf-lora-targets attention,mlp \
  --elf-lora-last-n-blocks 4 \
  --dascoli-simulated-source \
  --dascoli-train-accuracies 0.25 0.50 0.75 0.90 \
  --sentence-source-eval-json "$t5_source" \
  --sentence-source-eval-confidence 0.5 \
  --sentence-source-copy-confidence-threshold "$copy_threshold" \
  --sentence-source-flow-end-time "$flow_end_time" \
  --dascoli-eval-mode joint_weighted \
  --dascoli-eval-semantic-scale "$semantic_scale" \
  --dascoli-confidence-floor 1.0 \
  --val-num-examples 110 \
  --eval-num-examples 110 \
  --retrieval-num-examples -1 \
  --batch_size 64 \
  --num_sampling_steps 32 \
  --target-latents-mode lazy \
  --target-latent-encode-batch-size 128 \
  --train-target-mask-mode full \
  --generation-t5-retrieval \
  --eval-only \
  --checkpoint-metric word_content_mean \
  --seed 49
