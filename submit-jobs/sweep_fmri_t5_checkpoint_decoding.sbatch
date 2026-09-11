#!/usr/bin/env bash
#SBATCH --job-name=fmri_t5sweep
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=submit-jobs/sweep_fmri_t5_checkpoint_decoding-%j.out
#SBATCH --error=submit-jobs/sweep_fmri_t5_checkpoint_decoding-%j.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=${PYTHON_BIN:-$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python}
checkpoint=${CHECKPOINT:?set CHECKPOINT to one bundled fMRI-to-T5 best.pt}
output_json=${OUTPUT_JSON:?set OUTPUT_JSON}
secondary_content_head=${SECONDARY_CONTENT_HEAD_CHECKPOINT:-}
secondary_content_mix=${SECONDARY_CONTENT_MIX:-0}

# Slurm uses commas to delimit --export entries.  Accept colons from the
# submission environment and normalize them only after the job starts.
topks=${TOPKS:-3,5,8}; topks=${topks//:/,}
bias_strengths=${BIAS_STRENGTHS:-1,2,3,4}; bias_strengths=${bias_strengths//:/,}
bias_once_values=${BIAS_ONCE_VALUES:-0,1}; bias_once_values=${bias_once_values//:/,}
bias_weightings=${BIAS_WEIGHTINGS:-ranked}; bias_weightings=${bias_weightings//:/,}
bias_temperatures=${BIAS_TEMPERATURES:-1}; bias_temperatures=${bias_temperatures//:/,}
prior_subtractions=${PRIOR_SUBTRACTIONS:-0.25,0.5,0.75}; prior_subtractions=${prior_subtractions//:/,}
max_prior_probabilities=${MAX_PRIOR_PROBABILITIES:-1}; max_prior_probabilities=${max_prior_probabilities//:/,}
keyword_contexts=${KEYWORD_CONTEXTS:-0,1}; keyword_contexts=${keyword_contexts//:/,}
keyword_strengths=${KEYWORD_STRENGTHS:-1}; keyword_strengths=${keyword_strengths//:/,}
keyword_templates=${KEYWORD_TEMPLATES:-labeled,raw,summarize,sentence}; keyword_templates=${keyword_templates//:/,}
min_generation_tokens=${MIN_GENERATION_TOKENS:-0,8,10}; min_generation_tokens=${min_generation_tokens//:/,}
repetition_penalties=${REPETITION_PENALTIES:-1}; repetition_penalties=${repetition_penalties//:/,}
no_repeat_ngram_sizes=${NO_REPEAT_NGRAM_SIZES:-0}; no_repeat_ngram_sizes=${no_repeat_ngram_sizes//:/,}

cd "$project_root"
export PYTHONPATH="$project_root:$project_root/src:$project_root/scripts:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$data_root/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
"$python_bin" scripts/sweep_fmri_t5_checkpoint_decoding.py \
  --checkpoint "$checkpoint" \
  --secondary-content-head-checkpoint "$secondary_content_head" \
  --secondary-content-mix "$secondary_content_mix" \
  --oof-semantic-npz "$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz" \
  --text-npz "$data_root/minilm/knowntext_unseenbrain_v1/oracle_exact_minilm384_train_val.npz" \
  --output-json "$output_json" \
  --topks "$topks" \
  --bias-strengths "$bias_strengths" \
  --bias-once-values "$bias_once_values" \
  --bias-weightings "$bias_weightings" \
  --bias-temperatures "$bias_temperatures" \
  --prior-subtractions "$prior_subtractions" \
  --max-prior-probabilities "$max_prior_probabilities" \
  --keyword-contexts "$keyword_contexts" \
  --keyword-strengths "$keyword_strengths" \
  --keyword-templates "$keyword_templates" \
  --min-generation-tokens "$min_generation_tokens" \
  --num-beams "${NUM_BEAMS:-2}" \
  --repetition-penalties "$repetition_penalties" \
  --no-repeat-ngram-sizes "$no_repeat_ngram_sizes" \
  --max-output-words "${MAX_OUTPUT_WORDS:-10}" \
  --derangements "${DERANGEMENTS:-5}" \
  --top-derangement-configs "${TOP_DERANGEMENT_CONFIGS:-8}" \
  --seed 49
