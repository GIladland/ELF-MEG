#!/usr/bin/env bash
#SBATCH --job-name=qc4_semword
#SBATCH --clusters=htc
#SBATCH --partition=interactive
#SBATCH --qos=standard
#SBATCH --account=engs-pnpl
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-2%1
#SBATCH --output=submit-jobs/train_qc4wyals_semantic_word_head_20260912-%A_%a.out
#SBATCH --error=submit-jobs/train_qc4wyals_semantic_word_head_20260912-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
runs_root=${RUNS_ROOT:-/data/engs-pnpl/glandau/elf-runs}
python_bin=$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python
view=$data_root/tang_themoth_qc4wyals_nested_oof_20260908/interface/qc4wyals_oof_wordhead_view_20260912.npz

task=${SLURM_ARRAY_TASK_ID:-0}
case "$task" in
  0)
    arm=linear_idf0_pair0p1
    hidden=0
    lr=3e-4
    idf=0.0
    pair=0.1
    ;;
  1)
    arm=mlp256_idf0p5_pair0p1
    hidden=256
    lr=1e-4
    idf=0.5
    pair=0.1
    ;;
  2)
    arm=mlp512_idf0p5_pair0p3
    hidden=512
    lr=1e-4
    idf=0.5
    pair=0.3
    ;;
  *)
    echo "Unsupported task $task" >&2
    exit 2
    ;;
esac

[[ -f "$view" ]] || { echo "Missing semantic word-head view: $view" >&2; exit 2; }
run_tag=meg_elfm_puresem_wordhead_${arm}_seed49_20260912
output_dir=$runs_root/$run_tag
mkdir -p "$output_dir" "$data_root/tmp"

export HF_HOME=$data_root/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export TMPDIR=$data_root/tmp
export PYTHONPATH=$project_root/src:$project_root${PYTHONPATH:+:$PYTHONPATH}

cd "$project_root"
exec "$python_bin" scripts/probe_fmri_supervised_content_head.py \
  --brain-npz "$view" \
  --text-npz "$view" \
  --semantic-npz "$view" \
  --semantic-delay-mode ordered \
  --checkpoint unused-in-semantic-mode \
  --output-json "$output_dir/metrics.json" \
  --train-count 2652 \
  --val-count 110 \
  --feature semantic \
  --head-hidden-dim "$hidden" \
  --prior-initialization \
  --min-frequency 2 \
  --max-vocabulary 3000 \
  --negative-topk 64 \
  --negative-weight 0.25 \
  --positive-idf-power "$idf" \
  --selection-metric f1 \
  --pairwise-positive-weight "$pair" \
  --epochs 40 \
  --batch-size 128 \
  --lr "$lr" \
  --weight-decay 1e-3 \
  --seed 49 \
  --device cuda
