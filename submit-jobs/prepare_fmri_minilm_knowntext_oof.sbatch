#!/usr/bin/env bash
#SBATCH --job-name=prep_fmri_known
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=submit-jobs/prepare_fmri_minilm_knowntext_oof-%j.out
#SBATCH --error=submit-jobs/prepare_fmri_minilm_knowntext_oof-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$DATA_ROOT/xdg-cache}
export HF_HOME=${HF_HOME:-$DATA_ROOT/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}

source_dir=$DATA_ROOT/minilm/oracle1536_delayed_seg10_middle10content_clockfix
oof_dir=$DATA_ROOT/minilm/oof_robustness
output_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
train_val_source=$source_dir/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_oracle1536_middle10content_clockfix_train_val.npz
test_source=$source_dir/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_oracle1536_middle10content_clockfix_test.npz
oof_source=$oof_dir/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz
train_val_exact=$output_dir/oracle_exact_minilm384_train_val.npz
test_exact=$output_dir/oracle_exact_minilm384_test107.npz
oof_exact_targets=$output_dir/oofmix_exact_minilm384_targets_train23450_val266.npz
all_exact=$output_dir/oracle_exact_minilm384_all12098_testfirst.npz

source "$(conda info --base)/etc/profile.d/conda.sh"
stamp_path=$TMP_ENV_ROOT/.elf_env_stamp
if [[ ! -d "$TMP_ENV_ROOT/bin" || ! -f "$stamp_path" || "$(cat "$stamp_path")" != "$ENV_STAMP" ]]; then
  bash "$PROJECT_ROOT/scripts/arc_setup_elf_env.sh"
fi
conda activate "$TMP_ENV_ROOT"
cd "$PROJECT_ROOT"
mkdir -p "$output_dir"

reembed() {
  local source_npz=$1
  local output_npz=$2
  if [[ -f "$output_npz" ]]; then
    echo "Reusing $output_npz"
    return
  fi
  python scripts/reembed_semantic_npz_sentences.py \
    --input-npz "$source_npz" \
    --output "$output_npz" \
    --embedding-model-name sentence-transformers/all-MiniLM-L6-v2 \
    --pooling mean \
    --max-length 128 \
    --batch-size 256 \
    --device cuda
}

reembed "$train_val_source" "$train_val_exact"
reembed "$test_source" "$test_exact"
reembed "$oof_source" "$oof_exact_targets"

# The 107 known test sentences are deliberately first. All 12,098 rows remain
# training rows; same-row generation monitoring therefore measures the known-text
# oracle ceiling and never consumes an MRI2SEM-predicted test vector.
if [[ ! -f "$all_exact" ]]; then
  python scripts/merge_semantic_npzs.py \
    --input-npz "$test_exact" \
    --input-npz "$train_val_exact" \
    --output "$all_exact"
fi

echo "Prepared known-text/OFF artifacts in $output_dir"

