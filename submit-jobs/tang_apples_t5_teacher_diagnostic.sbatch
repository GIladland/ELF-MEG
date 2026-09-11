#!/usr/bin/env bash
#SBATCH --job-name=ta_t5teacher
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/tang_apples_t5_teacher_diagnostic-%j.out
#SBATCH --error=submit-jobs/tang_apples_t5_teacher_diagnostic-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
source_dir=$data_root/tang_themoth_apples_exact_ada002_textmid_seg10_v1
source_npz=$source_dir/tang_apples_ada002_knowntext_all2788_testfirst.npz
latent_cache=$source_dir/tang_apples_ada002_knowntext_all2788_t5small_latents_fp16.npy
teacher_npz=$source_dir/tang_apples_ada002_knowntext_all2788_t5context.npz
target_length=${TEACHER_TARGET_LENGTH:-27}
if [[ "${TEACHER_LENGTH_SWEEP:-0}" == "1" ]]; then
  lengths=(12 16 20 24)
  target_length=${lengths[${SLURM_ARRAY_TASK_ID:-0}]}
fi
output_suffix=${TEACHER_OUTPUT_SUFFIX:-}
if [[ -n "$output_suffix" ]]; then
  output_suffix=_$output_suffix
fi
output_dir=/data/engs-pnpl/glandau/elf-runs/tang_apples_t5_teacher_frozen_diagnostic_len${target_length}${output_suffix}_20260901
env_root=$data_root/conda-envs/braindiffusion-elf-torch271cu118

if ! command -v conda >/dev/null 2>&1; then
  source /apps/system/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$env_root"
export XDG_CACHE_HOME=$data_root/xdg-cache
export HF_HOME=$data_root/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TOKENIZERS_PARALLELISM=false
cd "$project_root"

if [[ ! -f "$teacher_npz" ]]; then
  python scripts/export_text_t5_context_npz.py \
    --source-npz "$source_npz" \
    --target-latents-cache "$latent_cache" \
    --output "$teacher_npz"
fi

# The protected test texts are deliberately first in this known-text oracle
# pack.  This diagnostic never loads a MEG-predicted vector.
python scripts/decode_elf_context_npz.py \
  --npz-path "$teacher_npz" \
  --output-dir "$output_dir" \
  --context-key context \
  --context-mask-key context_mask \
  --target-latents-key target_t5_latents \
  --target-mask-key t5_attention_mask \
  --target-ids-key t5_input_ids \
  --sentence-key sentence \
  --num-examples 26 \
  --target-length "$target_length" \
  --batch-size 13 \
  --num-sampling-steps 32 \
  --cfg-scale 1.0 \
  --self-cond-cfg-scale 1.0 \
  --decode-oracle-target \
  --retrieval-num-examples 26 \
  --retrieval-batch-size 128 \
  --device cuda
