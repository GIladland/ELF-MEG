#!/usr/bin/env bash
#SBATCH --job-name=test_semadapt
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=submit-jobs/test_semantic_adapter_arc-%j.out
#SBATCH --error=submit-jobs/test_semantic_adapter_arc-%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
ENV_ROOT=${ENV_ROOT:-/data/engs-pnpl/glandau/elf-cache/conda-envs/braindiffusion-elf-torch271cu118}

cd "$PROJECT_ROOT"
"$ENV_ROOT/bin/python" tests/test_semantic_adapter.py
"$ENV_ROOT/bin/python" tests/test_fmri2sem_bridge.py
"$ENV_ROOT/bin/python" tests/test_elf_lora.py
"$ENV_ROOT/bin/python" tests/test_content_token_weighting.py
"$ENV_ROOT/bin/python" tests/test_content_bow_coverage_loss.py
"$ENV_ROOT/bin/python" tests/test_add_bertscore_to_metrics.py
"$ENV_ROOT/bin/python" tests/test_collect_fmri_long_program_metrics.py
"$ENV_ROOT/bin/python" tests/test_semantic_alignment_interface.py
"$ENV_ROOT/bin/python" tests/test_semantic_preservation_loss.py
"$ENV_ROOT/bin/python" -m py_compile \
  scripts/meg_context_overfit.py \
  scripts/train_npz_semantic_to_elf.py \
  scripts/add_bertscore_to_metrics.py \
  src/modules/lora.py \
  src/modules/fmri2sem_bridge.py \
  src/modules/semantic_adapter.py
"$ENV_ROOT/bin/python" scripts/train_npz_semantic_to_elf.py --help >/dev/null
