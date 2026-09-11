#!/usr/bin/env bash
#SBATCH --job-name=fmri_elf_pkg
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --qos=standard
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/package_fmri_elf_content_winner-%j.out
#SBATCH --error=submit-jobs/package_fmri_elf_content_winner-%j.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
environment_root=${TMP_ENV_ROOT:-$data_root/conda-envs/braindiffusion-elf-torch271cu118}
export PATH="$environment_root/bin:$PATH"
export PYTHONNOUSERSITE=1
cd "$project_root"

winner_root=/data/engs-pnpl/glandau/elf-runs/fmri_minilm_elf_xattn_last2_lr1e4_from_recover2_step1000_20260831_lexgate_top5_frompartstep37_top10_prior0p1_t2_scattered_b40_val266_generationonly
source_checkpoint=/data/engs-pnpl/glandau/elf-runs/fmri_minilm_elf_xattn_last2_lr1e4_from_recover2_step1000_20260831_lexgate_top5_partitioned_dev0p1eval64_fixrepeat4/final.pt
lexical_checkpoint=/data/engs-pnpl/glandau/elf-runs/fmri_supervised_content_probe_20260831/semantic1536_mlp512_priorresidual.pt

python scripts/package_fmri_elf_lexical_diffusion.py \
  --checkpoint "$source_checkpoint" \
  --lexical-checkpoint "$lexical_checkpoint" \
  --metrics-json "$winner_root/eval_step_000000_t5retrieval.json" \
  --deranged-metrics \
    "${winner_root}_roll1/eval_step_000000.json" \
    "${winner_root}_roll17/eval_step_000000.json" \
    "${winner_root}_roll83/eval_step_000000.json" \
  --bertscore-json "$winner_root/eval_step_000000_bertscore.json" \
  --output "$winner_root/packaged_system/fmri_elf_content_first_val266_20260831.pt" \
  --name fmri_elf_content_first_val266_20260831 \
  --lexical-topk 10 \
  --lexical-temperature 2.0 \
  --lexical-prior-subtraction 0.1 \
  --lexical-max-prior 1.0 \
  --lexical-bias-strength 40 \
  --lexical-bias-mode scattered \
  --lexical-context-mode partitioned
