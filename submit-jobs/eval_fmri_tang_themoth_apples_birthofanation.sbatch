#!/usr/bin/env bash
#SBATCH --job-name=eval_fmri_tm_apples
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=submit-jobs/eval_fmri_tang_themoth_apples_birthofanation-%j.out
#SBATCH --error=submit-jobs/eval_fmri_tang_themoth_apples_birthofanation-%j.err

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
subset_dir=$DATA_ROOT/tang_themoth_apples_v1/fmri_birthofanation_subset
source_run=fmri_minilm_knowntext_oofmix1536_resmlp_h512_norm_lr5e4_inputonly_ep20_b32_seed49_20260829
prior_eval=eval_${source_run}_unseenbrain_pred1536_decodeseed101_test107_20260829

source_npz=$DATA_ROOT/minilm/oracle1536_delayed_seg10_middle10content_clockfix/uts01_minilm_l6v2_delayed_d1-2-3-4_seg10_pred1536_test_middle10content_clockfix.npz
target_npz=$known_dir/oracle_exact_minilm384_test107.npz
source_subset=$subset_dir/pred1536_birthofanation26.npz
target_subset=$subset_dir/oracle_exact_minilm384_birthofanation26.npz
full_latents=$known_dir/eval_latents/${prior_eval}_t5small_latents_fp16.npy
subset_latents=$subset_dir/birthofanation26_t5small_latents_fp16.npy

mkdir -p "$subset_dir"
cd "$PROJECT_ROOT"

python scripts/subset_npz_by_story.py \
  --input-npz "$source_npz" \
  --output-npz "$source_subset" \
  --story birthofanation \
  --overwrite

python scripts/subset_npz_by_story.py \
  --input-npz "$target_npz" \
  --output-npz "$target_subset" \
  --story birthofanation \
  --parallel-npy "$full_latents" \
  --output-parallel-npy "$subset_latents" \
  --overwrite

python - "$source_subset" "$target_subset" <<'PY'
import sys
import numpy as np

with np.load(sys.argv[1], allow_pickle=True) as source, np.load(sys.argv[2], allow_pickle=True) as target:
    assert len(source["sentence"]) == len(target["sentence"]) == 26
    assert np.array_equal(source["story"].astype(str), target["story"].astype(str))
    assert np.array_equal(source["start_tr"], target["start_tr"])
    assert np.array_equal(source["stop_tr"], target["stop_tr"])
    assert np.array_equal(source["sentence"].astype(str), target["sentence"].astype(str))
    assert source["input_embeddings"].shape == (26, 1536)
    assert target["input_embeddings"].shape == (26, 384)
print("Verified matched birthofanation source/target rows: 26")
PY

export NPZ_PATH=$source_subset
export SOURCE_NPZ=$NPZ_PATH
export RUN_TAG=eval_fmri_tang_themoth_APPLES_birthofanation26_fixedelf_steps16_cfg0p5_seed101_20260830
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export TARGET_LATENTS_CACHE=$subset_latents
export INIT_E2E_CHECKPOINT=/data/engs-pnpl/glandau/elf-runs/$source_run/best.pt
export INIT_E2E_ADAPTER_MISMATCH=error
export SEMANTIC_ALIGNMENT_TARGET_NPZ=$target_subset
export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings

export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export NUM_EXAMPLES=26
export VAL_NUM_EXAMPLES=0
export EVAL_NUM_EXAMPLES=26
export RETRIEVAL_NUM_EXAMPLES=26
export RETRIEVAL_BATCH_SIZE=64
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export SEMANTIC_ADAPTER_KIND=residual_mlp
export SEMANTIC_MAPPER_HIDDEN_DIM=512
export SEMANTIC_DELAY_TOKENS=4
export SEMANTIC_INPUT_PROJECTION_DIM=0
export NORMALIZE_SEMANTIC_PROJECTION=1
export FREEZE_ELF=1
export EVAL_ONLY=1
export NUM_SAMPLING_STEPS=16
export CFG_SCALE=0.5
export SELF_COND_CFG_SCALE=1.0
export SEED=101
export USE_WANDB=1
export WANDB_RUN_GROUP=tang_themoth_apples_fmri2text_20260830
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Matched apples-to-apples fMRI control: locked birthofanation story only, 26 ten-word windows, same 26-text candidate bank and fixed 16-step/CFG-0.5/seed-101 sampler as the MEG route. Brain model/checkpoint remains fixed."
export SAVE_FINAL_CHECKPOINT=0

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
