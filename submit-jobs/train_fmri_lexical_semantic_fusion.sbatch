#!/usr/bin/env bash
#SBATCH --job-name=fmri_lexfusion
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/train_fmri_lexical_semantic_fusion-%j.out
#SBATCH --error=submit-jobs/train_fmri_lexical_semantic_fusion-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
mri_run=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1
direct_run=/data/engs-pnpl/glandau/elf-runs/fmri_minilm_stage2_direct384_semanticonly_exactalign_from_delaytok_best_ep100_seed49_20260830_v100shortprobe_from_best_single_step1000_20260831
probe_dir=/data/engs-pnpl/glandau/elf-runs/fmri_supervised_content_probe_20260831
output_dir=/data/engs-pnpl/glandau/elf-runs/fmri_lexical_semantic_fusion_20260831

cd "$project_root"
export PYTHONPATH="$project_root/src:$project_root/scripts:${PYTHONPATH:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$data_root/conda-envs/braindiffusion-elf-torch271cu118"

python scripts/train_fmri_lexical_semantic_fusion.py \
  --brain-npz "$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz" \
  --text-npz "$data_root/minilm/knowntext_unseenbrain_v1/oracle_exact_minilm384_train_val.npz" \
  --mri-checkpoint "$direct_run/semantic_retrieval_step250_snapshot.pt" \
  --head-checkpoint "$probe_dir/semantic384_directstep250_mlp512_priorresidual.pt" \
  --output-json "$output_dir/direct384_top5_residual512_val266.json" \
  --export-npz "$output_dir/direct384_top5_residual512_train_val.npz" \
  --mri-output-dim 384 \
  --lexical-topk 5 \
  --lexical-temperature 0.5 \
  --fusion-hidden-dim 512 \
  --epochs 50 \
  --batch-size 256 \
  --lr 3e-4
