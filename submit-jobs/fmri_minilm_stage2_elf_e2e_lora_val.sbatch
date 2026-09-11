#!/usr/bin/env bash
#SBATCH --job-name=fmri_e2e_elf
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-38
#SBATCH --output=submit-jobs/fmri_minilm_stage2_elf_e2e_lora_val-%A_%a.out
#SBATCH --error=submit-jobs/fmri_minilm_stage2_elf_e2e_lora_val-%A_%a.err

set -euo pipefail

# Stage 2 starts from the validation-selected raw-fMRI delay-token baseline.
# Only train_x/val_x are loaded. The 107-row brain test split remains closed.
export PROJECT_ROOT=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
export DATA_ROOT=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
known_dir=$DATA_ROOT/minilm/knowntext_unseenbrain_v1
mri_run=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1
baseline_run=fmri_minilm_rawbrain_e2e_projector_long100_resdelaytok_h1024_l2_h8_storybal_b32_seed49_20260830
baseline_checkpoint=/data/engs-pnpl/glandau/elf-runs/$baseline_run/best.pt
stage2_init_checkpoint=${STAGE2_INIT_CHECKPOINT_OVERRIDE:-$baseline_checkpoint}

export FREEZE_ELF=0
export LAST_N_BLOCKS=-1
export ELF_LORA_RANK=0
export ELF_LORA_ALPHA=16
export ELF_LORA_DROPOUT=0.05
export ELF_LORA_TARGETS=attention
export ELF_LORA_LAST_N_BLOCKS=-1
export ELF_LR=2e-6
export BATCH_SIZE=16
export BRAIN_LEXICAL_HEAD_CHECKPOINT=
export BRAIN_LEXICAL_TOPK=5
export BRAIN_LEXICAL_TEMPERATURE=0.5
export BRAIN_LEXICAL_PRIOR_SUBTRACTION=0.0
export BRAIN_LEXICAL_DECODE_BIAS_STRENGTH=0.0
content_focus=${CONTENT_FOCUS:-0}
stage_mapper_lr=5e-5
stage_brain_lr=5e-6
stage_brain_unfreeze=projector
stage_semantic_alignment_weight=1.0
stage_semantic_contrastive_weight=0.1
stage_semantic_context_weight=0.1
stage_train_semantic_input_only=1
stage_preserve_delay_context=0
stage_init_adapter_mismatch=error
stage_brain_semantic_output_dim=1536
stage_brain_projector_mismatch=error
stage_semantic_adapter_kind=residual_delay_tokens
stage_checkpoint_metric=content_word_overlap
stage_eval_every=1000
stage_warmup_steps=1000

case "$SLURM_ARRAY_TASK_ID" in
  0)
    variant=full_elf
    ;;
  1)
    variant=lora_attn_r8
    export FREEZE_ELF=1
    export ELF_LORA_RANK=8
    export ELF_LORA_ALPHA=16
    export ELF_LR=1e-5
    export BATCH_SIZE=32
    ;;
  2)
    variant=lora_attn_mlp_r8
    export FREEZE_ELF=1
    export ELF_LORA_RANK=8
    export ELF_LORA_ALPHA=16
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LR=1e-5
    export BATCH_SIZE=32
    ;;
  3)
    variant=inputonly_frozenelf_contentw2_bow1p0_stdlr
    export FREEZE_ELF=1
    export ELF_LR=0
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=1.0
    export DECODER_LOSS_WEIGHT=1.5
    export COND_DROPOUT_PROB=0.0
    ;;
  4)
    variant=inputonly_frozenelf_contentw2_bow1p0_lowlr
    export FREEZE_ELF=1
    export ELF_LR=0
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=1.0
    export DECODER_LOSS_WEIGHT=1.5
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=1e-5
    stage_brain_lr=1e-6
    ;;
  5)
    variant=inputonly_frozenelf_contentw2_bow2p0_lowlr
    export FREEZE_ELF=1
    export ELF_LR=0
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=2.0
    export DECODER_LOSS_WEIGHT=1.5
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=1e-5
    stage_brain_lr=1e-6
    ;;
  6)
    variant=inputonly_frozenelf_contentw2_bow1p0_ultralowlr
    export FREEZE_ELF=1
    export ELF_LR=0
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=1.0
    export DECODER_LOSS_WEIGHT=1.5
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=2e-6
    stage_brain_lr=2e-7
    ;;
  7)
    variant=inputonly_frozenelf_contentw2_bow1p0_midlr
    export FREEZE_ELF=1
    export ELF_LR=0
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=1.0
    export DECODER_LOSS_WEIGHT=1.5
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=5e-6
    stage_brain_lr=5e-7
    ;;
  8)
    variant=inputonly_frozenelf_contentw2_bow0p5_lowlr
    export FREEZE_ELF=1
    export ELF_LR=0
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.5
    export DECODER_LOSS_WEIGHT=1.5
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=1e-5
    stage_brain_lr=1e-6
    ;;
  9)
    variant=inputonly_frozenelf_contentw3_bow1p0_midlr_dec2
    export FREEZE_ELF=1
    export ELF_LR=0
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=3.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=1.0
    export DECODER_LOSS_WEIGHT=2.0
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=5e-6
    stage_brain_lr=5e-7
    ;;
  10)
    variant=inputonly_frozenelf_contentw2_bow1p0_midlr_nosemaux
    export FREEZE_ELF=1
    export ELF_LR=0
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=1.0
    export DECODER_LOSS_WEIGHT=1.5
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=5e-6
    stage_brain_lr=5e-7
    stage_semantic_alignment_weight=0.0
    stage_semantic_contrastive_weight=0.0
    stage_semantic_context_weight=0.0
    ;;
  11)
    # Same lexical objective as the provisional winner, with a 2.5x smaller
    # input-side LR to reduce the rapid semantic/lexical drift after step 1k.
    variant=inputonly_frozenelf_contentw3_bow1p0_ultralowlr_dec2
    export FREEZE_ELF=1
    export ELF_LR=0
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=3.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=1.0
    export DECODER_LOSS_WEIGHT=2.0
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=2e-6
    stage_brain_lr=2e-7
    ;;
  12)
    # Retain strong position-wise content supervision but halve the
    # order-invariant coverage term to test BERT/WER recovery.
    variant=inputonly_frozenelf_contentw3_bow0p5_midlr_dec2
    export FREEZE_ELF=1
    export ELF_LR=0
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=3.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.5
    export DECODER_LOSS_WEIGHT=2.0
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=5e-6
    stage_brain_lr=5e-7
    ;;
  13)
    # Keep both content losses but reduce the total decoder coefficient, which
    # gives the semantic alignment/context losses more relative influence.
    variant=inputonly_frozenelf_contentw3_bow1p0_midlr_dec1p5
    export FREEZE_ELF=1
    export ELF_LR=0
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=3.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=1.0
    export DECODER_LOSS_WEIGHT=1.5
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=5e-6
    stage_brain_lr=5e-7
    ;;
  14)
    # Recovery from the content-specialist checkpoint: a very small rank-4
    # update in only the last two ELF blocks, with mild lexical supervision.
    variant=recover_lora_last2_r4_mildcontent
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=1e-6
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=1.5
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.0
    export DECODER_LOSS_WEIGHT=1.0
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=1e-6
    stage_brain_lr=1e-7
    ;;
  15)
    # Same narrow LoRA update while explicitly preserving the content gain.
    variant=recover_lora_last2_r4_content
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=1e-6
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=3.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.5
    export DECODER_LOSS_WEIGHT=1.5
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=1e-6
    stage_brain_lr=1e-7
    ;;
  16)
    # No ELF update: train the complete MiniLM-to-context projector at a very
    # small LR, to compare representational adaptation against narrow LoRA.
    variant=recover_fulladapter_ultralowlr_content
    export FREEZE_ELF=1
    export ELF_LR=0
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=3.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.5
    export DECODER_LOSS_WEIGHT=1.5
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=1e-6
    stage_brain_lr=1e-7
    stage_train_semantic_input_only=0
    ;;
  17)
    # Ultra-local recovery from the content-LoRA peak. The reduced update and
    # 250-step validation cadence search the narrow content/BERT trade-off.
    variant=recover2_lora_last2_r4_semantic_content
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=2e-7
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.25
    export DECODER_LOSS_WEIGHT=1.0
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=2e-7
    stage_brain_lr=2e-8
    stage_semantic_alignment_weight=2.0
    stage_semantic_contrastive_weight=0.2
    stage_semantic_context_weight=0.2
    stage_eval_every=250
    stage_warmup_steps=250
    ;;
  18)
    # Strong semantic/BERT recovery with minimal lexical pressure. Starting
    # content margin is large enough to tolerate a small lexical regression.
    variant=recover2_lora_last2_r4_semantic_strong
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=2e-7
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=1.5
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.0
    export DECODER_LOSS_WEIGHT=1.0
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=2e-7
    stage_brain_lr=2e-8
    stage_semantic_alignment_weight=4.0
    stage_semantic_contrastive_weight=0.4
    stage_semantic_context_weight=0.4
    stage_eval_every=250
    stage_warmup_steps=250
    ;;
  19)
    # Bidirectional lexical objective: retain target coverage while applying a
    # very mild penalty to likely content tokens absent from this row.
    variant=recover3_lora_last2_r4_precision0p05
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=2e-7
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=2.5
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.25
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.05
    export DECODER_CONTENT_PRECISION_TOPK=32
    export DECODER_LOSS_WEIGHT=1.5
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=2e-7
    stage_brain_lr=2e-8
    stage_semantic_alignment_weight=4.0
    stage_semantic_contrastive_weight=0.4
    stage_semantic_context_weight=0.4
    stage_eval_every=250
    stage_warmup_steps=250
    ;;
  20)
    # Balanced precision arm.
    variant=recover3_lora_last2_r4_precision0p15
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=2e-7
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=2.5
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.25
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.15
    export DECODER_CONTENT_PRECISION_TOPK=32
    export DECODER_LOSS_WEIGHT=1.5
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=2e-7
    stage_brain_lr=2e-8
    stage_semantic_alignment_weight=4.0
    stage_semantic_contrastive_weight=0.4
    stage_semantic_context_weight=0.4
    stage_eval_every=250
    stage_warmup_steps=250
    ;;
  21)
    # Strong precision arm to test whether frequent-token collapse is the
    # dominant failure mode rather than insufficient semantic conditioning.
    variant=recover3_lora_last2_r4_precision0p30
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=2e-7
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=2.5
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.25
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.30
    export DECODER_CONTENT_PRECISION_TOPK=32
    export DECODER_LOSS_WEIGHT=1.5
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=2e-7
    stage_brain_lr=2e-8
    stage_semantic_alignment_weight=4.0
    stage_semantic_contrastive_weight=0.4
    stage_semantic_context_weight=0.4
    stage_eval_every=250
    stage_warmup_steps=250
    ;;
  22)
    # Sequence-recovery arm: stronger positional CE and easier decoder noise,
    # while retaining moderate content coverage and precision pressure.
    variant=recover3_lora_last2_r4_seqrecover_precision0p15
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=2e-7
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.25
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.15
    export DECODER_CONTENT_PRECISION_TOPK=32
    export DECODER_LOSS_WEIGHT=2.5
    export DECODER_NOISE_SCALE=1.5
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=2e-7
    stage_brain_lr=2e-8
    stage_semantic_alignment_weight=2.0
    stage_semantic_contrastive_weight=0.2
    stage_semantic_context_weight=0.2
    stage_eval_every=250
    stage_warmup_steps=250
    ;;
  23)
    # Direct train-only lexical ranking on the predicted 384-D semantic vector.
    variant=recover3_lora_last2_r4_semcontent0p10
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=2e-7
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=1.5
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.0
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.0
    export DECODER_LOSS_WEIGHT=1.0
    export SEMANTIC_CONTENT_LOSS_WEIGHT=0.10
    export SEMANTIC_CONTENT_NEGATIVE_TOPK=32
    export SEMANTIC_CONTENT_TEMPERATURE=0.1
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=2e-7
    stage_brain_lr=2e-8
    stage_semantic_alignment_weight=4.0
    stage_semantic_contrastive_weight=0.4
    stage_semantic_context_weight=0.4
    stage_eval_every=250
    stage_warmup_steps=250
    ;;
  24)
    variant=recover3_lora_last2_r4_semcontent0p50
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=2e-7
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=1.5
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.0
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.0
    export DECODER_LOSS_WEIGHT=1.0
    export SEMANTIC_CONTENT_LOSS_WEIGHT=0.50
    export SEMANTIC_CONTENT_NEGATIVE_TOPK=32
    export SEMANTIC_CONTENT_TEMPERATURE=0.1
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=2e-7
    stage_brain_lr=2e-8
    stage_semantic_alignment_weight=4.0
    stage_semantic_contrastive_weight=0.4
    stage_semantic_context_weight=0.4
    stage_eval_every=250
    stage_warmup_steps=250
    ;;
  25)
    variant=recover3_lora_last2_r4_semcontent1p50
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=2e-7
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=1.5
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.0
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.0
    export DECODER_LOSS_WEIGHT=1.0
    export SEMANTIC_CONTENT_LOSS_WEIGHT=1.50
    export SEMANTIC_CONTENT_NEGATIVE_TOPK=32
    export SEMANTIC_CONTENT_TEMPERATURE=0.1
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=2e-7
    stage_brain_lr=2e-8
    stage_semantic_alignment_weight=4.0
    stage_semantic_contrastive_weight=0.4
    stage_semantic_context_weight=0.4
    stage_eval_every=250
    stage_warmup_steps=250
    ;;
  26)
    # Combine mild decoder-side precision with direct semantic lexical ranking.
    variant=recover3_lora_last2_r4_semcontent0p50_precision0p05
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=2e-7
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=2.5
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.25
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.05
    export DECODER_CONTENT_PRECISION_TOPK=32
    export DECODER_LOSS_WEIGHT=1.5
    export SEMANTIC_CONTENT_LOSS_WEIGHT=0.50
    export SEMANTIC_CONTENT_NEGATIVE_TOPK=32
    export SEMANTIC_CONTENT_TEMPERATURE=0.1
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=2e-7
    stage_brain_lr=2e-8
    stage_semantic_alignment_weight=4.0
    stage_semantic_contrastive_weight=0.4
    stage_semantic_context_weight=0.4
    stage_eval_every=250
    stage_warmup_steps=250
    ;;
  27)
    # Fast semantic-interface pretraining: keep the complete generator fixed
    # and optimize the MRI projector/ordered-delay mapper for lexical ranking.
    variant=semanticonly_semcontent0p50_lowlr
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=2e-7
    export BATCH_SIZE=32
    export SEMANTIC_OBJECTIVE_ONLY=1
    export SEMANTIC_CONTENT_LOSS_WEIGHT=0.50
    export SEMANTIC_CONTENT_NEGATIVE_TOPK=32
    export SEMANTIC_CONTENT_TEMPERATURE=0.1
    stage_mapper_lr=2e-6
    stage_brain_lr=2e-7
    stage_semantic_alignment_weight=4.0
    stage_semantic_contrastive_weight=0.4
    stage_semantic_context_weight=0.4
    stage_eval_every=250
    stage_warmup_steps=100
    ;;
  28)
    variant=semanticonly_semcontent1p50_lowlr
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=2e-7
    export BATCH_SIZE=32
    export SEMANTIC_OBJECTIVE_ONLY=1
    export SEMANTIC_CONTENT_LOSS_WEIGHT=1.50
    export SEMANTIC_CONTENT_NEGATIVE_TOPK=32
    export SEMANTIC_CONTENT_TEMPERATURE=0.1
    stage_mapper_lr=2e-6
    stage_brain_lr=2e-7
    stage_semantic_alignment_weight=4.0
    stage_semantic_contrastive_weight=0.4
    stage_semantic_context_weight=0.4
    stage_eval_every=250
    stage_warmup_steps=100
    ;;
  29)
    variant=semanticonly_semcontent1p50_midlr
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=2e-7
    export BATCH_SIZE=32
    export SEMANTIC_OBJECTIVE_ONLY=1
    export SEMANTIC_CONTENT_LOSS_WEIGHT=1.50
    export SEMANTIC_CONTENT_NEGATIVE_TOPK=32
    export SEMANTIC_CONTENT_TEMPERATURE=0.1
    stage_mapper_lr=1e-5
    stage_brain_lr=1e-6
    stage_semantic_alignment_weight=4.0
    stage_semantic_contrastive_weight=0.4
    stage_semantic_context_weight=0.4
    stage_eval_every=250
    stage_warmup_steps=100
    ;;
  30)
    # Preserve the four ordered MRI2SEM delays as explicit residual ELF
    # conditioning slots while keeping the proven 384-D context path intact.
    # The new branch is zero-initialized, so step zero exactly matches the
    # validation-selected single-checkpoint baseline.
    variant=recover4_delaycontext_lora_last2_r4
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=2e-7
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=1.5
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.0
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.0
    export DECODER_LOSS_WEIGHT=1.0
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=2e-7
    stage_brain_lr=2e-8
    stage_semantic_alignment_weight=4.0
    stage_semantic_contrastive_weight=0.4
    # Oracle target context has no delay residual; do not train the new path
    # to collapse back to it through context MSE.
    stage_semantic_context_weight=0.0
    stage_preserve_delay_context=1
    stage_init_adapter_mismatch=delay-context
    stage_eval_every=250
    stage_warmup_steps=250
    ;;
  31)
    # Same explicit-delay interface plus direct train-only lexical ranking on
    # the fused 384-D semantic output.
    variant=recover4_delaycontext_lora_last2_r4_semcontent0p50
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=2e-7
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=1.5
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.0
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.0
    export DECODER_LOSS_WEIGHT=1.0
    export SEMANTIC_CONTENT_LOSS_WEIGHT=0.50
    export SEMANTIC_CONTENT_NEGATIVE_TOPK=32
    export SEMANTIC_CONTENT_TEMPERATURE=0.1
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=2e-7
    stage_brain_lr=2e-8
    stage_semantic_alignment_weight=4.0
    stage_semantic_contrastive_weight=0.4
    stage_semantic_context_weight=0.0
    stage_preserve_delay_context=1
    stage_init_adapter_mismatch=delay-context
    stage_eval_every=250
    stage_warmup_steps=250
    ;;
  32)
    # Correct the target contract directly: average the four pretrained
    # 384-D output heads into one exact-sentence head, then train that head
    # against the exact ten-word MiniLM target. ELF remains unchanged.
    variant=direct384_semanticonly_exactalign
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=2e-7
    export BATCH_SIZE=32
    export SEMANTIC_OBJECTIVE_ONLY=1
    export SEMANTIC_CONTENT_LOSS_WEIGHT=0.0
    stage_mapper_lr=2e-6
    stage_brain_lr=1e-5
    stage_semantic_alignment_weight=4.0
    stage_semantic_contrastive_weight=0.4
    stage_semantic_context_weight=0.4
    stage_brain_semantic_output_dim=384
    stage_brain_projector_mismatch=mean-blocks
    stage_semantic_adapter_kind=residual_identity
    stage_init_adapter_mismatch=direct-identity
    stage_checkpoint_metric=semantic_top5
    stage_eval_every=250
    stage_warmup_steps=100
    ;;
  33)
    # Same direct 384-D target correction with train-only lexical ranking.
    variant=direct384_semanticonly_semcontent0p50
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=2e-7
    export BATCH_SIZE=32
    export SEMANTIC_OBJECTIVE_ONLY=1
    export SEMANTIC_CONTENT_LOSS_WEIGHT=0.50
    export SEMANTIC_CONTENT_NEGATIVE_TOPK=32
    export SEMANTIC_CONTENT_TEMPERATURE=0.1
    stage_mapper_lr=2e-6
    stage_brain_lr=1e-5
    stage_semantic_alignment_weight=4.0
    stage_semantic_contrastive_weight=0.4
    stage_semantic_context_weight=0.4
    stage_brain_semantic_output_dim=384
    stage_brain_projector_mismatch=mean-blocks
    stage_semantic_adapter_kind=residual_identity
    stage_init_adapter_mismatch=direct-identity
    stage_checkpoint_metric=semantic_top5
    stage_eval_every=250
    stage_warmup_steps=100
    ;;
  34)
    # Convert the validation-improved direct-384 semantic head into text while
    # protecting it with strong semantic losses and ultra-small E2E updates.
    variant=direct384_recover_lora_last2_r4_semantic
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=2e-7
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=1.5
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.0
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.0
    export DECODER_LOSS_WEIGHT=1.0
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=2e-7
    stage_brain_lr=2e-8
    stage_semantic_alignment_weight=4.0
    stage_semantic_contrastive_weight=0.4
    stage_semantic_context_weight=0.4
    stage_brain_semantic_output_dim=384
    stage_brain_projector_mismatch=error
    stage_semantic_adapter_kind=residual_identity
    stage_init_adapter_mismatch=error
    stage_checkpoint_metric=content_word_overlap
    stage_eval_every=250
    stage_warmup_steps=250
    ;;
  35)
    # Content-first direct-384 recovery.  Positional CE remains active, while
    # order-invariant coverage supplies a modest lexical gradient through ELF.
    variant=direct384_recover_lora_last2_r4_content
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=2e-7
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=3.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.5
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.0
    export DECODER_LOSS_WEIGHT=1.5
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=2e-7
    stage_brain_lr=2e-8
    stage_semantic_alignment_weight=4.0
    stage_semantic_contrastive_weight=0.4
    stage_semantic_context_weight=0.4
    stage_brain_semantic_output_dim=384
    stage_brain_projector_mismatch=error
    stage_semantic_adapter_kind=residual_identity
    stage_init_adapter_mismatch=error
    stage_checkpoint_metric=content_word_overlap
    stage_eval_every=250
    stage_warmup_steps=250
    ;;
  36)
    # Freeze the newly improved brain head and adapt only the residual context
    # path plus narrow LoRA.  This isolates ELF calibration from brain drift.
    variant=direct384_recover_lora_last2_r4_frozenbrain
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=2e-7
    export BATCH_SIZE=32
    export DECODER_CONTENT_TOKEN_WEIGHT=2.0
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.25
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.0
    export DECODER_LOSS_WEIGHT=1.0
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=2e-7
    stage_brain_lr=0.0
    stage_brain_unfreeze=frozen
    stage_semantic_alignment_weight=4.0
    stage_semantic_contrastive_weight=0.4
    stage_semantic_context_weight=0.4
    stage_brain_semantic_output_dim=384
    stage_brain_projector_mismatch=error
    stage_semantic_adapter_kind=residual_identity
    stage_init_adapter_mismatch=error
    stage_checkpoint_metric=content_word_overlap
    stage_eval_every=250
    stage_warmup_steps=250
    ;;
  37)
    # Test whether the nonlinear information visible in the frozen 2048-D
    # lexical probe can improve the exact-384 semantic contract.  This is a
    # semantic-only screen: the final residual MRI block and projector move
    # at a very small LR while the proven ELF/context net stays frozen.
    variant=direct384_semanticonly_lastblock_exactalign
    export FREEZE_ELF=1
    # The direct-step250 initialization contains the baseline rank-4 LoRA
    # tensors.  Recreate them for strict checkpoint compatibility; the
    # semantic-only objective supplies no ELF gradients.
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=0
    export BATCH_SIZE=32
    export SEMANTIC_OBJECTIVE_ONLY=1
    export SEMANTIC_CONTENT_LOSS_WEIGHT=0.0
    stage_mapper_lr=2e-7
    stage_brain_lr=2e-7
    stage_brain_unfreeze=last_block
    stage_semantic_alignment_weight=4.0
    stage_semantic_contrastive_weight=0.4
    stage_semantic_context_weight=0.4
    stage_brain_semantic_output_dim=384
    stage_brain_projector_mismatch=error
    stage_semantic_adapter_kind=residual_identity
    stage_init_adapter_mismatch=error
    stage_checkpoint_metric=semantic_top5
    stage_eval_every=250
    stage_warmup_steps=100
    ;;
  38)
    # Keep the improved direct semantic interface intact and supply the
    # held-out lexical head as a separate gated ELF context.  The gate starts
    # at exactly zero, so this branch is checkpoint-identical at step zero.
    if [[ -z "${STAGE2_INIT_CHECKPOINT_OVERRIDE:-}" ]]; then
      stage2_init_checkpoint=/data/engs-pnpl/glandau/elf-runs/fmri_minilm_stage2_direct384_semanticonly_exactalign_from_delaytok_best_ep100_seed49_20260830_v100shortprobe_from_best_single_step1000_20260831/semantic_retrieval_step250_snapshot.pt
    fi
    variant=direct384_recover_lora_last2_r4_lexicalctx_top5
    export FREEZE_ELF=1
    export ELF_LORA_RANK=4
    export ELF_LORA_ALPHA=8
    export ELF_LORA_TARGETS=attention,mlp
    export ELF_LORA_LAST_N_BLOCKS=2
    export ELF_LR=2e-7
    export BATCH_SIZE=32
    export BRAIN_LEXICAL_HEAD_CHECKPOINT=${BRAIN_LEXICAL_HEAD_CHECKPOINT_OVERRIDE:-/data/engs-pnpl/glandau/elf-runs/fmri_supervised_content_probe_20260831/semantic384_directstep250_mlp512_priorresidual.pt}
    export BRAIN_LEXICAL_TOPK=${BRAIN_LEXICAL_TOPK_OVERRIDE:-5}
    export BRAIN_LEXICAL_TEMPERATURE=${BRAIN_LEXICAL_TEMPERATURE_OVERRIDE:-0.5}
    export BRAIN_LEXICAL_PRIOR_SUBTRACTION=${BRAIN_LEXICAL_PRIOR_SUBTRACTION_OVERRIDE:-0.0}
    export BRAIN_LEXICAL_DECODE_BIAS_STRENGTH=${BRAIN_LEXICAL_DECODE_BIAS_STRENGTH_OVERRIDE:-0.0}
    export DECODER_CONTENT_TOKEN_WEIGHT=1.5
    export DECODER_CONTENT_BOW_LOSS_WEIGHT=0.1
    export DECODER_CONTENT_PRECISION_LOSS_WEIGHT=0.0
    export DECODER_LOSS_WEIGHT=1.0
    export COND_DROPOUT_PROB=0.0
    stage_mapper_lr=2e-7
    stage_brain_lr=0.0
    stage_brain_unfreeze=frozen
    stage_semantic_alignment_weight=4.0
    stage_semantic_contrastive_weight=0.4
    stage_semantic_context_weight=0.4
    stage_brain_semantic_output_dim=384
    stage_brain_projector_mismatch=error
    stage_semantic_adapter_kind=residual_identity
    stage_init_adapter_mismatch=error
    stage_checkpoint_metric=content_word_overlap
    stage_eval_every=${EVAL_EVERY_OVERRIDE:-250}
    stage_warmup_steps=${WARMUP_STEPS_OVERRIDE:-250}
    ;;
  *)
    echo "Unsupported array task: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

if [[ "$content_focus" == "1" ]]; then
  variant=contentw3_dec2_${variant}
  export DECODER_CONTENT_TOKEN_WEIGHT=3.0
  export DECODER_LOSS_WEIGHT=2.0
  export COND_DROPOUT_PROB=0.0
fi

export RUN_TAG=fmri_minilm_stage2_${variant}_from_delaytok_best_ep100_seed49_20260830${STAGE2_RUN_SUFFIX:-}
export OUTPUT_DIR=/data/engs-pnpl/glandau/elf-runs/$RUN_TAG
export NPZ_PATH=$known_dir/oracle_exact_minilm384_train_val.npz
export SOURCE_NPZ=$NPZ_PATH
export INPUT_KEY=input_embeddings
export SENTENCE_KEY=sentence
export TRAIN_BALANCE_KEY=story

# Load both the trained MRI projector and the complete adapter/ELF state from
# the stage-2 baseline. This is continuation, not reinitialization.
export BRAIN_INPUT_NPZ=$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz
export BRAIN_TRAIN_KEY=train_x
export BRAIN_VAL_KEY=val_x
export BRAIN_MODEL_CHECKPOINT=$stage2_init_checkpoint
export BRAIN_SEMANTIC_OUTPUT_DIM=$stage_brain_semantic_output_dim
export BRAIN_PROJECTOR_MISMATCH=$stage_brain_projector_mismatch
export BRAIN_HIDDEN_DIM=2048
export BRAIN_RES_BLOCKS=4
export BRAIN_DROPOUT=0.1
export BRAIN_UNFREEZE=$stage_brain_unfreeze
export BRAIN_LR=$stage_brain_lr

export SEMANTIC_ADAPTER_KIND=$stage_semantic_adapter_kind
export SEMANTIC_MAPPER_HIDDEN_DIM=1024
export SEMANTIC_DELAY_TOKENS=4
export SEMANTIC_DELAY_LAYERS=2
export SEMANTIC_DELAY_HEADS=8
export ADAPTER_DROPOUT=0.1
export SEMANTIC_INPUT_PROJECTION_DIM=0
export TRAIN_SEMANTIC_INPUT_ONLY=$stage_train_semantic_input_only
export NORMALIZE_SEMANTIC_PROJECTION=0
export SEMANTIC_PRESERVE_DELAY_CONTEXT=$stage_preserve_delay_context
export INIT_E2E_CHECKPOINT=$stage2_init_checkpoint
export INIT_E2E_ADAPTER_MISMATCH=$stage_init_adapter_mismatch

export SEMANTIC_ALIGNMENT_TARGET_NPZ=${SEMANTIC_ALIGNMENT_TARGET_NPZ_OVERRIDE:-$known_dir/oracle_exact_minilm384_train_val.npz}
export ALLOW_SEMANTIC_ALIGNMENT_SENTENCE_MISMATCH=${ALLOW_SEMANTIC_ALIGNMENT_SENTENCE_MISMATCH:-0}
if [[ -n "${SEMANTIC_ALIGNMENT_TARGET_NPZ_OVERRIDE:-}" ]]; then
  export ALLOW_SEMANTIC_ALIGNMENT_SENTENCE_MISMATCH=1
fi
export SEMANTIC_ALIGNMENT_TARGET_KEY=input_embeddings
export SEMANTIC_ALIGNMENT_LOSS_WEIGHT=$stage_semantic_alignment_weight
export SEMANTIC_ALIGNMENT_LOSS_TYPE=cosine
export SEMANTIC_CONTRASTIVE_LOSS_WEIGHT=$stage_semantic_contrastive_weight
export SEMANTIC_CONTRASTIVE_TEMPERATURE=0.07
export SEMANTIC_CONTEXT_LOSS_WEIGHT=$stage_semantic_context_weight

export TARGET_LATENTS_CACHE=$known_dir/oracle_exact_minilm384_trainval_current10_t5small_latents_fp16.npy
export TMP_ENV_ROOT=${TMP_ENV_ROOT:-$DATA_ROOT/conda-envs/braindiffusion-elf-torch271cu118}
export ENV_STAMP=${ENV_STAMP:-braindiffusion-elf-torch271cu118-py310-v1}
export SKIP_ARC_SETUP=1
export NUM_EXAMPLES=0
export VAL_NUM_EXAMPLES=266
export EVAL_NUM_EXAMPLES=${EVAL_NUM_EXAMPLES_OVERRIDE:-266}
export RETRIEVAL_NUM_EXAMPLES=${RETRIEVAL_NUM_EXAMPLES_OVERRIDE:-$EVAL_NUM_EXAMPLES}
export EVAL_EVERY=$stage_eval_every
export RETRIEVAL_EVAL_EVERY=$stage_eval_every
export RETRIEVAL_BATCH_SIZE=256
export EPOCHS=100
export CONTEXT_LENGTH=64
export SEMANTIC_HIDDEN_DIM=4096
export LR=$stage_mapper_lr
export WEIGHT_DECAY=0.001
export LR_SCHEDULE=cosine
export WARMUP_STEPS=$stage_warmup_steps
export MIN_LR_RATIO=0.01
export COND_DROPOUT_PROB=${COND_DROPOUT_PROB:-0.05}
export NUM_SAMPLING_STEPS=32
export TRAIN_TIMESTEP_STEPS=32
export CHECKPOINT_METRIC=$stage_checkpoint_metric
export SAVE_FINAL_CHECKPOINT=1
export SAVE_EVAL_CHECKPOINTS=1
export EVAL_CHECKPOINT_TOP_K=1
export EVAL_CHECKPOINT_EVERY=0
export SEED=49
export USE_WANDB=${USE_WANDB_OVERRIDE:-1}
export WANDB_RUN_GROUP=fmri_minilm_stage2_elf_e2e_lora_val_20260830
export WANDB_RUN_NAME=$RUN_TAG
export WANDB_NOTES="Stage-2 raw-fMRI E2E continuation from $stage2_init_checkpoint. Variant=$variant. MRI projector LR=$stage_brain_lr residual delay-token mapper LR=$stage_mapper_lr; optional ELF/LoRA update. CONTENT_FOCUS=$content_focus decoder_content_token_weight=${DECODER_CONTENT_TOKEN_WEIGHT:-1.0} decoder_content_bow_loss_weight=${DECODER_CONTENT_BOW_LOSS_WEIGHT:-0.0} decoder_content_precision_loss_weight=${DECODER_CONTENT_PRECISION_LOSS_WEIGHT:-0.0} semantic_content_loss_weight=${SEMANTIC_CONTENT_LOSS_WEIGHT:-0.0} decoder_loss_weight=${DECODER_LOSS_WEIGHT:-1.0}. Train11725/val266 only; test107 brain is never loaded."

exec bash "$PROJECT_ROOT/submit-jobs/tang_gpt_podcast_short_overfit_npz_to_elf.sbatch"
