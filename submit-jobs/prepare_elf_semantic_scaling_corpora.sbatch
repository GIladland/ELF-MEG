#!/usr/bin/env bash
#SBATCH --job-name=elf_scale_prep
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-1%2
#SBATCH --output=submit-jobs/prepare_elf_semantic_scaling_corpora-%A_%a.out
#SBATCH --error=submit-jobs/prepare_elf_semantic_scaling_corpora-%A_%a.err

set -euo pipefail

project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=${PYTHON_BIN:-$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python}
scale_root=$data_root/semantic_scaling_20260902
mkdir -p "$scale_root"

export HF_HOME=${HF_HOME:-$data_root/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    modality=mri_minilm
    exact=$data_root/minilm/knowntext_unseenbrain_v1/oracle_exact_minilm384_all12098_testfirst.npz
    audit=$data_root/minilm/knowntext_unseenbrain_v1/oracle_exact_minilm384_train_val.npz
    audit_tail=266
    max_aug=84686
    small_aug=36294
    large_aug=84686
    small_scale=x4
    large_scale=x8
    small_replay=4
    large_replay=8
    ;;
  1)
    modality=meg_ada
    exact=$data_root/tang_themoth_apples_exact_ada002_textmid_seg10_v1/tang_apples_ada002_knowntext_all2788_testfirst.npz
    audit=$data_root/tang_themoth_qc4wyals_ada_v2/interface/qc4wyals_val_exact_ada002.npz
    audit_tail=0
    max_aug=41820
    small_aug=8364
    large_aug=41820
    small_scale=x4
    large_scale=x16
    small_replay=4
    large_replay=16
    ;;
  *) exit 2 ;;
esac

work=$scale_root/$modality
mkdir -p "$work"
minilm_aug=$work/same_corpus_phrasebank_${max_aug}_minilm384.npz
if [[ ! -f "$minilm_aug" ]]; then
  "$python_bin" "$project_root/scripts/augment_sherlock_sva_recombinational.py" \
    --train-npz "$exact" \
    --output "$minilm_aug" \
    --mode phrasebank \
    --target-count "$max_aug" \
    --seed 20260902 \
    --min-words 7 \
    --max-words 13 \
    --filter-mode simple_sv \
    --embedding-model-name sentence-transformers/all-MiniLM-L6-v2 \
    --batch-size 512 \
    --device "${AUG_DEVICE:-cuda}" \
    --local-files-only
fi

if [[ "$modality" == meg_ada ]]; then
  augmentation=$work/same_corpus_phrasebank_${max_aug}_ada002.npz
  key_file=${OPENAI_API_KEY_FILE:-/data/engs-pnpl/glandau/BrainDiffusion/secrets/openai_api_key}
  if [[ -z "${OPENAI_API_KEY:-}" && -f "$key_file" ]]; then
    export OPENAI_API_KEY
    OPENAI_API_KEY="$(tr -d '\r\n' < "$key_file")"
  fi
  if [[ ! -f "$augmentation" ]]; then
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      echo "Missing OPENAI_API_KEY for ADA augmentation embedding" >&2
      exit 3
    fi
    "$python_bin" "$project_root/scripts/reembed_semantic_npz_openai.py" \
      --input-npz "$minilm_aug" \
      --output "$augmentation" \
      --model text-embedding-ada-002 \
      --batch-size 256
  fi
else
  augmentation=$minilm_aug
fi

for scale in "$small_scale" "$large_scale"; do
  if [[ "$scale" == "$small_scale" ]]; then
    aug_limit=$small_aug
    anchor_repeats=$small_replay
  else
    aug_limit=$large_aug
    anchor_repeats=$large_replay
  fi
  output=$work/${modality}_${scale}_anchorbalanced_auditfirst.npz
  "$python_bin" "$project_root/scripts/build_nested_semantic_scaling_pack.py" \
    --exact-npz "$exact" \
    --augmentation-npz "$augmentation" \
    --audit-npz "$audit" \
    --audit-tail "$audit_tail" \
    --augmentation-limit "$aug_limit" \
    --anchor-repeats "$anchor_repeats" \
    --label "${modality}_${scale}_anchorbalanced" \
    --output "$output"
done

echo "prepared modality=$modality root=$work"
