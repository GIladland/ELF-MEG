#!/usr/bin/env bash
#SBATCH --job-name=gut1m_prep
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=submit-jobs/gutenberg_detective_nosherlock_selected1m_prep-%j.out
#SBATCH --error=submit-jobs/gutenberg_detective_nosherlock_selected1m_prep-%j.err

set -euo pipefail

export PREP_ONLY=1
export GUTENBERG_CANDIDATE_OUTPUT=${GUTENBERG_CANDIDATE_OUTPUT:-/data/engs-pnpl/glandau/elf-cache/gutenberg/gutenberg_detective_nosherlock_candidates2m_allbooks_minilm_seed7.npz}
export GUTENBERG_SELECTED_OUTPUT=${GUTENBERG_SELECTED_OUTPUT:-/data/engs-pnpl/glandau/elf-cache/gutenberg/gutenberg_detective_nosherlock_selected1m_by_sherlock1to9_minilm_seed7.npz}
export GUTENBERG_SELECTED_COUNT=${GUTENBERG_SELECTED_COUNT:-1000000}
export GUTENBERG_MAX_BOOKS=${GUTENBERG_MAX_BOOKS:-0}
export GUTENBERG_LIMIT_SENTENCES=${GUTENBERG_LIMIT_SENTENCES:-2000000}
export GUTENBERG_QUERY_SAMPLE_SIZE=${GUTENBERG_QUERY_SAMPLE_SIZE:-8192}
export GUTENBERG_SCORE_MODE=${GUTENBERG_SCORE_MODE:-max}
export COMBINED_OUTPUT=${COMBINED_OUTPUT:-/data/engs-pnpl/glandau/elf-cache/gutenberg/gutenberg_detective_nosherlock_selected1m_train_sherlock12_val_minilm_seed7.npz}
export INCLUDE_SHERLOCK_TRAIN=${INCLUDE_SHERLOCK_TRAIN:-0}
export SKIP_ARC_SETUP=${SKIP_ARC_SETUP:-1}
export SELECTION_REPORT_OUTPUT=${SELECTION_REPORT_OUTPUT:-/data/engs-pnpl/glandau/elf-cache/gutenberg/gutenberg_detective_nosherlock_selected1m_by_sherlock1to9_minilm_seed7.book_report.json}
export SELECTION_REPORT_TOP_BOOKS=${SELECTION_REPORT_TOP_BOOKS:-200}
export SELECTION_REPORT_EXAMPLES_PER_BOOK=${SELECTION_REPORT_EXAMPLES_PER_BOOK:-3}
export WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
export WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
export WANDB_DIR=${WANDB_DIR:-/data/engs-pnpl/glandau/elf-cache/wandb}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-/data/engs-pnpl/glandau/elf-cache/wandb-data}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/data/engs-pnpl/glandau/elf-cache/wandb-cache}
export TMPDIR=${TMPDIR:-/data/engs-pnpl/glandau/elf-cache/tmp}

bash ./scripts/arc_run_minilm_gutenberg_holdout.sh
