#!/usr/bin/env bash
#SBATCH --job-name=fmri_time_targets
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=priority
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=submit-jobs/prepare_fmri_temporally_aligned_minilm_targets-%j.out
#SBATCH --error=submit-jobs/prepare_fmri_temporally_aligned_minilm_targets-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
mri_root=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM
mri_run=$mri_root/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1
base_dir=$data_root/minilm/oracle1536_delayed_seg10
base_prefix=uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_oracle1536
target_root=$data_root/minilm/temporally_aligned_targets_20260831
stimulus_root=/data/engs-pnpl/glandau/semantic-decoding
env_root=${TMP_ENV_ROOT:-$data_root/conda-envs/braindiffusion-elf-torch271cu118}
target_device=${TARGET_DEVICE:-cuda}

export PYTHONPATH="$project_root/src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export HF_HOME=${HF_HOME:-$data_root/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$env_root"
cd "$project_root"
mkdir -p "$target_root"

prepare_variant() {
  local variant=$1
  shift
  local output_dir=$target_root/$variant
  local prefix=uts01_minilm_l6v2_${variant}
  mkdir -p "$output_dir"
  python scripts/relabel_mri2sem_oracle_elf_middle_words.py \
    --input-dir "$base_dir" \
    --input-prefix "$base_prefix" \
    --output-dir "$output_dir" \
    --output-prefix "$prefix" \
    --metadata-json "$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.json" \
    --stimulus-data-root "$stimulus_root" \
    --mri2sem-src "$mri_root/src" \
    --expected-dim 1536 \
    --expected-train 11725 \
    --expected-val 266 \
    --expected-test 107 \
    "$@"
  python scripts/reembed_semantic_npz_sentences.py \
    --input-npz "$output_dir/${prefix}_train_val.npz" \
    --output "$output_dir/${prefix}_train_val_minilm384.npz" \
    --embedding-model-name sentence-transformers/all-MiniLM-L6-v2 \
    --pooling mean \
    --max-length 128 \
    --batch-size 256 \
    --device "$target_device" \
    --local-files-only
}

# Same ten-word evaluation length, but centered on the brain interval rather
# than selected anywhere in +/-10 seconds for maximum content density.
prepare_variant centered10 --word-count 10

# Every word whose midpoint lies in the complete 20-second brain interval.
# This is the temporally faithful semantic auxiliary; it is not the decoder
# target and therefore does not change the fixed content-rich val metric.
prepare_variant full20s --time-window-radius-sec 10

echo "prepared temporally aligned train11725/val266 targets in $target_root; no predicted test brain vectors loaded"
