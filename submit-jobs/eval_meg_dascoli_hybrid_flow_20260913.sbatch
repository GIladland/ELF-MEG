#!/usr/bin/env bash
#SBATCH --job-name=meg_dascolih
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-5%4
#SBATCH --output=submit-jobs/eval_meg_dascoli_hybrid_flow_20260913-%A_%a.out
#SBATCH --error=submit-jobs/eval_meg_dascoli_hybrid_flow_20260913-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
runs_root=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
tmp_env_root=${TMP_ENV_ROOT:-/tmp/${USER}-braindiffusion-elf-torch213}
env_stamp=${ENV_STAMP:-braindiffusion-elf-torch213-py310-20260803}
interface=$data_root/tang_themoth_qc4wyals_nested_oof_20260908/interface
source_npz=$interface/qc4wyals_oof_predicted_train_valtail_ada002.npz
dascoli_source=$project_root/meg_t5_elf_gap_closure_20260912/dascoli_initial_topprior_0.01_top10_minmax_global.json
selected_checkpoint=${SELECTED_CHECKPOINT:-$runs_root/qc4wyals_elfb_dascoli_lora8_mid_ep40_seed49_20260911/best.pt}

# Keep these low-to-high and map 0-5 to 6 semantic scales.
semantic_scales=(0.0 0.02 0.05 0.1 0.25 1.0)
semantic_scale=${semantic_scales[${SLURM_ARRAY_TASK_ID:-0}]}
run_tag=qc4wyals_elfb_dascolisource_hybrid_sem${semantic_scale}_seed49_20260913
output_dir=$runs_root/$run_tag

for required in "$source_npz" "$dascoli_source" "$selected_checkpoint"; do
  [[ -f "$required" ]] || { echo "Missing required path: $required" >&2; exit 2; }
done
mkdir -p "$output_dir" "$data_root/tmp"
export HF_HOME=$data_root/huggingface TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=$data_root/tmp

cd "$project_root"
PROJECT_ROOT="$project_root" DATA_ROOT="$data_root" TMP_ENV_ROOT="$tmp_env_root" ENV_STAMP="$env_stamp" \
  bash "$project_root/scripts/arc_setup_elf_env.sh"
python_bin=$tmp_env_root/bin/python

[[ -x "$python_bin" ]] || { echo "Missing python binary: $python_bin" >&2; exit 2; }

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
  --sentence-source-eval-json "$dascoli_source" \
  --sentence-source-eval-confidence 0.5 \
  --sentence-source-copy-confidence-threshold -1.0 \
  --dascoli-eval-mode joint_weighted \
  --dascoli-eval-semantic-scale "$semantic_scale" \
  --dascoli-confidence-floor 0.00 \
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
