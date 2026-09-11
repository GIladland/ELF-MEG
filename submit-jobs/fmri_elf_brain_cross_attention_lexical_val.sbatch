#!/usr/bin/env bash
#SBATCH --job-name=fmri_xattn_lex
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-3
#SBATCH --output=submit-jobs/fmri_elf_brain_cross_attention_lexical_val-%A_%a.out
#SBATCH --error=submit-jobs/fmri_elf_brain_cross_attention_lexical_val-%A_%a.err

set -euo pipefail

# Follow-up to the plain cross-attention screen.  The lexical probe was fit on
# train11725 only and has held-out val266 top-5 content F1 0.05769.  Its frozen
# prototypes enter through a zero-init 64-value gate; selected arms also apply
# the same brain-conditioned token bias during ELF's final diffusion decode.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
lexical_head=/data/engs-pnpl/glandau/elf-runs/fmri_supervised_content_probe_20260831/semantic1536_mlp512_priorresidual.pt

export BRAIN_LEXICAL_HEAD_CHECKPOINT=$lexical_head
export BRAIN_LEXICAL_TEMPERATURE=0.5
export BRAIN_LEXICAL_PRIOR_SUBTRACTION=0.0
export BRAIN_LEXICAL_DECODE_BIAS_STRENGTH=0.0
export BRAIN_LEXICAL_MAX_PRIOR_PROBABILITY=1.0
export BRAIN_LEXICAL_DECODE_BIAS_ONCE=0
export BRAIN_LEXICAL_DECODE_BIAS_MODE=sequence
export TRAIN_LEXICAL_CONTEXT_GATE_ONLY=1
export ELF_BRAIN_CROSS_ATTENTION_DECODER_ONLY=1
export DENOISER_LOSS_WEIGHT_OVERRIDE=0.0

case "$SLURM_ARRAY_TASK_ID" in
  0)
    export BRAIN_LEXICAL_TOPK=5
    export CROSS_RUN_SUFFIX=_lexgate_top5
    ;;
  1)
    export BRAIN_LEXICAL_TOPK=8
    export CROSS_RUN_SUFFIX=_lexgate_top8
    ;;
  2)
    export BRAIN_LEXICAL_TOPK=5
    export BRAIN_LEXICAL_PRIOR_SUBTRACTION=0.25
    export BRAIN_LEXICAL_DECODE_BIAS_STRENGTH=0.5
    export CROSS_RUN_SUFFIX=_lexgate_top5_prior0p25_bias0p5
    ;;
  3)
    export BRAIN_LEXICAL_TOPK=8
    export BRAIN_LEXICAL_PRIOR_SUBTRACTION=0.25
    export BRAIN_LEXICAL_DECODE_BIAS_STRENGTH=1.0
    export CROSS_RUN_SUFFIX=_lexgate_top8_prior0p25_bias1
    ;;
  *)
    echo "Unsupported task $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

export BRAIN_LEXICAL_TOPK=${BRAIN_LEXICAL_TOPK_OVERRIDE:-$BRAIN_LEXICAL_TOPK}
export BRAIN_LEXICAL_TEMPERATURE=${BRAIN_LEXICAL_TEMPERATURE_OVERRIDE:-$BRAIN_LEXICAL_TEMPERATURE}
export BRAIN_LEXICAL_PRIOR_SUBTRACTION=${BRAIN_LEXICAL_PRIOR_SUBTRACTION_OVERRIDE:-$BRAIN_LEXICAL_PRIOR_SUBTRACTION}
export BRAIN_LEXICAL_DECODE_BIAS_STRENGTH=${BRAIN_LEXICAL_DECODE_BIAS_STRENGTH_OVERRIDE:-$BRAIN_LEXICAL_DECODE_BIAS_STRENGTH}
export BRAIN_LEXICAL_MAX_PRIOR_PROBABILITY=${BRAIN_LEXICAL_MAX_PRIOR_PROBABILITY_OVERRIDE:-$BRAIN_LEXICAL_MAX_PRIOR_PROBABILITY}
export BRAIN_LEXICAL_DECODE_BIAS_ONCE=${BRAIN_LEXICAL_DECODE_BIAS_ONCE_OVERRIDE:-$BRAIN_LEXICAL_DECODE_BIAS_ONCE}
export BRAIN_LEXICAL_DECODE_BIAS_MODE=${BRAIN_LEXICAL_DECODE_BIAS_MODE_OVERRIDE:-$BRAIN_LEXICAL_DECODE_BIAS_MODE}
export CROSS_RUN_SUFFIX=${CROSS_RUN_SUFFIX}${LEXICAL_EXTRA_SUFFIX:-}

export WANDB_RUN_GROUP=fmri_elf_brain_cross_attention_lexical_val_20260831
export WANDB_NOTES="Leakage-safe ELF lexical-memory decoder-only cross-attention follow-up. Lexical head fit on train11725 only; val266 and test107 absent from head fitting. ELF diffusion trajectory remains frozen; zero-init lexical gate and explicit target-to-brain attention train only through the decoder objective. test107 never loaded. Promotion requires matched brain > fixed derangements and content F1 > 0.043743."

exec bash "$PROJECT_ROOT/submit-jobs/fmri_elf_brain_cross_attention_val.sbatch"
