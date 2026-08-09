#!/usr/bin/env bash
#SBATCH --job-name=gutcat1m_prep
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=260G
#SBATCH --output=submit-jobs/gutenberg_catalog_en_short5to18_balanced_selected1m_prep-%j.out
#SBATCH --error=submit-jobs/gutenberg_catalog_en_short5to18_balanced_selected1m_prep-%j.err

set -euo pipefail

export MODE=${MODE:-gutenberg_plus_sherlock}
export PREP_ONLY=${PREP_ONLY:-1}
export GUTENBERG_CATALOG_CSV_URL=${GUTENBERG_CATALOG_CSV_URL:-https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz}
export GUTENBERG_SHUFFLE_BOOKS=${GUTENBERG_SHUFFLE_BOOKS:-1}
export GUTENBERG_MAX_BOOKS=${GUTENBERG_MAX_BOOKS:-0}
export GUTENBERG_LIMIT_SENTENCES=${GUTENBERG_LIMIT_SENTENCES:-3000000}
export GUTENBERG_REQUEST_TIMEOUT=${GUTENBERG_REQUEST_TIMEOUT:-60}
export GUTENBERG_RETRIES=${GUTENBERG_RETRIES:-1}
export GUTENBERG_DOWNLOAD_DELAY=${GUTENBERG_DOWNLOAD_DELAY:-0.02}
export GUTENBERG_QUERY_SAMPLE_SIZE=${GUTENBERG_QUERY_SAMPLE_SIZE:-0}
export GUTENBERG_SELECTION_MODE=${GUTENBERG_SELECTION_MODE:-query_balanced}
export GUTENBERG_PER_QUERY_CANDIDATES=${GUTENBERG_PER_QUERY_CANDIDATES:-80}
export GUTENBERG_SELECTED_COUNT=${GUTENBERG_SELECTED_COUNT:-1000000}
export GUTENBERG_CANDIDATE_FULL_OUTPUT=${GUTENBERG_CANDIDATE_FULL_OUTPUT:-/data/engs-pnpl/glandau/elf-cache/gutenberg/gutenberg_catalog_en_short5to18_raw3m_minilm_seed7.npz}
export GUTENBERG_CANDIDATE_SHORT_OUTPUT=${GUTENBERG_CANDIDATE_SHORT_OUTPUT:-/data/engs-pnpl/glandau/elf-cache/gutenberg/gutenberg_catalog_en_short5to18_content3m_minilm_seed7.npz}
export GUTENBERG_SELECTED_OUTPUT=${GUTENBERG_SELECTED_OUTPUT:-/data/engs-pnpl/glandau/elf-cache/gutenberg/gutenberg_catalog_en_short5to18_querybalanced_selected1m_by_sherlock1to9_short5to18_minilm_seed7.npz}
export COMBINED_OUTPUT=${COMBINED_OUTPUT:-/data/engs-pnpl/glandau/elf-cache/gutenberg/sherlock_plus_gutenberg_catalog_en_short5to18_querybalanced_selected1m_train_sherlock12_short5to18_val_minilm_seed7.npz}
export OUTPUT_DIR=${OUTPUT_DIR:-/data/engs-pnpl/glandau/elf-runs/prep_gutenberg_catalog_en_short5to18_querybalanced_selected1m}
export SKIP_ARC_SETUP=${SKIP_ARC_SETUP:-1}
export WANDB_ENTITY=${WANDB_ENTITY:-giladland-university-of-oxford}
export WANDB_PROJECT=${WANDB_PROJECT:-BrainDiffusion}
export WANDB_DIR=${WANDB_DIR:-/data/engs-pnpl/glandau/elf-cache/wandb}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-/data/engs-pnpl/glandau/elf-cache/wandb-data}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/data/engs-pnpl/glandau/elf-cache/wandb-cache}
export TMPDIR=${TMPDIR:-/data/engs-pnpl/glandau/elf-cache/tmp}

bash ./scripts/arc_run_minilm_short_holdout.sh
