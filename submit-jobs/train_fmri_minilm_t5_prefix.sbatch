#!/usr/bin/env bash
#SBATCH --job-name=fmri_t5prefix
#SBATCH --clusters=htc
#SBATCH --partition=long
#SBATCH --qos=standard
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=submit-jobs/train_fmri_minilm_t5_prefix-%A_%a.out
#SBATCH --error=submit-jobs/train_fmri_minilm_t5_prefix-%A_%a.err

set -euo pipefail
project_root=${PROJECT_ROOT:-/data/engs-pnpl/glandau/BrainDiffusion/ELF}
data_root=${DATA_ROOT:-/data/engs-pnpl/glandau/elf-cache}
python_bin=${PYTHON_BIN:-$data_root/conda-envs/braindiffusion-elf-torch271cu118/bin/python}
mri_run=/data/engs-pnpl/glandau/fMRI-decoding/MRI2SEM/runs/uts01_minilm_l6v2_tr2s_delayed_d1-2-3-4_seg10_mean_lags4_stride2_30ep_64g4cpu_priority_v1
checkpoint=/data/engs-pnpl/glandau/elf-runs/fmri_minilm_stage2_direct384_semanticonly_exactalign_from_delaytok_best_ep100_seed49_20260830_v100shortprobe_from_best_single_step1000_20260831/semantic_retrieval_step250_snapshot.pt
task=${SLURM_ARRAY_TASK_ID:-0}
model_name=t5-small
prefix_length=16
hidden=1024
calibrator_hidden=0
brain_calibrator_kind=residual_mlp
semantic_alignment=0
semantic_contrastive=0
brain_lr=3e-5
brain_unfreeze_projector=0
brain_projector_lr=3e-6
oof_semantic=
oracle_semantic_npz=
oof_delay_mode=mean
semantic_warmup_epochs=0
semantic_warmup_lr=3e-4
content_head=
content_bias_topk=5
content_bias_strength=0
content_bias_once=0
t5_lora_rank=0
t5_lora_alpha=8
projector_kind=mlp
content_keyword_context=0
content_keyword_train_context=0
oracle_keyword_train_context=0
content_keyword_strength=1
content_keyword_template=labeled
content_token_ce_weight=1
content_idf_power=0
content_idf_max_multiplier=3
pairing_margin_weight=0
pairing_margin=0.25
case "$task" in
  0) variant=prefix16_h1024 ;;
  1) variant=prefix8_h512; prefix_length=8; hidden=512 ;;
  2) variant=prefix16_h1024_calibrator512; calibrator_hidden=512; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=1e-4 ;;
  3) variant=prefix16_h1024_oofmean_calibrator512; calibrator_hidden=512; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=1e-4; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz ;;
  4) variant=prefix16_h1024_oofmean_projector; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz ;;
  5) variant=prefix8_h512_oofmean_calibrator512; prefix_length=8; hidden=512; calibrator_hidden=512; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=1e-4; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz ;;
  6) variant=prefix16_h1024_oofordered_calibrator512; calibrator_hidden=512; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=1e-4; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered ;;
  7) variant=prefix16_h1024_oofordered_calibrator1024; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=1e-4; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered ;;
  8) variant=prefix16_h1024_oofordered_calibrator1024_semwarm; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=20 ;;
  9) variant=prefix16_h1024_oofmean_calibrator512_semwarm; calibrator_hidden=512; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; semantic_warmup_epochs=20 ;;
  10) variant=prefix16_oofordered_semwarm_lexk3_b0p5; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=20; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=3; content_bias_strength=0.5 ;;
  11) variant=prefix16_oofordered_semwarm_lexk5_b0p5; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=20; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=0.5 ;;
  12) variant=prefix16_oofordered_semwarm_lexk5_b1p0; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=20; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=1.0 ;;
  13) variant=prefix16_oofordered_semwarm_lexk8_b1p0; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=20; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=8; content_bias_strength=1.0 ;;
  14) variant=prefix16_oofordered_semwarm_lexk5_b2p0; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=20; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=2.0 ;;
  15) variant=prefix16_oofordered_semwarm_lora4; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=20; t5_lora_rank=4 ;;
  16) variant=prefix16_oofordered_semwarm_lora4_lexk5_b0p5; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=20; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=0.5; t5_lora_rank=4 ;;
  17) variant=prefix16_oofordered_semwarm_lora8_lexk5_b0p5; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=20; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=0.5; t5_lora_rank=8; t5_lora_alpha=16 ;;
  18) variant=prefix16_oofmean_projector_lexk5_b0p5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=0.5 ;;
  19) variant=prefix8_oofmean_projector_lexk5_b0p5; prefix_length=8; hidden=512; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=0.5 ;;
  20) variant=delaytransformer16_oofordered; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; projector_kind=delay_transformer ;;
  21) variant=delaytransformer16_oofordered_lexk5_b0p5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; projector_kind=delay_transformer; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=0.5 ;;
  22) variant=delaytransformer16_oofordered_lexk5_b1p0; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; projector_kind=delay_transformer; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=1.0 ;;
  23) variant=prefix16_oofordered_semwarm_keyword3; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=20; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=3; content_keyword_context=1 ;;
  24) variant=prefix16_oofordered_semwarm_keyword5; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=20; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_keyword_context=1 ;;
  25) variant=prefix16_oofordered_semwarm_keyword5_lexb0p5; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=20; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=0.5; content_keyword_context=1 ;;
  26) variant=delaytransformer16_oofordered_keyword5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; projector_kind=delay_transformer; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_keyword_context=1 ;;
  27) variant=prefix16_oofordered_contentce2_pair0p1; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=10; content_token_ce_weight=2; pairing_margin_weight=0.1 ;;
  28) variant=prefix16_oofordered_contentce3_pair0p1; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=10; content_token_ce_weight=3; pairing_margin_weight=0.1 ;;
  29) variant=prefix16_oofordered_contentce2_pair0p3; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=10; content_token_ce_weight=2; pairing_margin_weight=0.3 ;;
  30) variant=prefix16_oofordered_contentce2_pair0p1_lexb2; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=10; content_token_ce_weight=2; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=2.0 ;;
  31) variant=prefix16_oofordered_contentce2_pair0p1_keyword5train; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=10; content_token_ce_weight=2; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_keyword_context=1; content_keyword_train_context=1 ;;
  32) variant=prefix16_oofordered_contentce2_pair0p1_keyword5train_lexb1; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=10; content_token_ce_weight=2; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=1.0; content_keyword_context=1; content_keyword_train_context=1 ;;
  33) variant=delaytransformer16_oofordered_contentce2_pair0p1; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; projector_kind=delay_transformer; content_token_ce_weight=2; pairing_margin_weight=0.1 ;;
  34) variant=prefix16_oofordered_contentce2_pair0p1_lora4; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=10; content_token_ce_weight=2; pairing_margin_weight=0.1; t5_lora_rank=4 ;;
  35) variant=prefix16_oofordered_full20lex_keyword5; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=10; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0_full20s_to_exact.pt; content_bias_topk=5; content_keyword_context=1 ;;
  36) variant=prefix16_oofordered_full20lex_keyword5train; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=10; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0_full20s_to_exact.pt; content_bias_topk=5; content_keyword_context=1; content_keyword_train_context=1 ;;
  37) variant=prefix16_oofordered_full20lex_lexb2; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=10; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0_full20s_to_exact.pt; content_bias_topk=5; content_bias_strength=2.0 ;;
  38) variant=prefix16_oofordered_full20lex_contentce2_pair0p1_keyword5train_lexb1; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=10; content_token_ce_weight=2; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0_full20s_to_exact.pt; content_bias_topk=5; content_bias_strength=1.0; content_keyword_context=1; content_keyword_train_context=1 ;;
  39) variant=prefix16_oofordered_nowarm_align0p25_contrast1_contentce2_pair0p1; calibrator_hidden=1024; semantic_alignment=0.25; semantic_contrastive=1.0; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=0; content_token_ce_weight=2; pairing_margin_weight=0.1 ;;
  40) variant=prefix16_oofordered_nowarm_contrast1_contentce2_pair0p1; calibrator_hidden=1024; semantic_alignment=0; semantic_contrastive=1.0; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=0; content_token_ce_weight=2; pairing_margin_weight=0.1 ;;
  41) variant=prefix16_oofordered_nowarm_align0p1_contrast2_contentce2_pair0p1; calibrator_hidden=1024; semantic_alignment=0.1; semantic_contrastive=2.0; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=0; content_token_ce_weight=2; pairing_margin_weight=0.1 ;;
  42) variant=prefix16_oofordered_nowarm_decoderonly_contentce2_pair0p1; calibrator_hidden=1024; semantic_alignment=0; semantic_contrastive=0; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=0; content_token_ce_weight=2; pairing_margin_weight=0.1 ;;
  43) variant=t5base_prefix16_oofordered_nowarm_contrast1_contentce2_pair0p1; model_name=t5-base; calibrator_hidden=1024; semantic_alignment=0; semantic_contrastive=1.0; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=0; content_token_ce_weight=2; pairing_margin_weight=0.1 ;;
  44) variant=t5base_prefix16_oofordered_nowarm_contentce2_pair0p1_keyword5train_lexb1; model_name=t5-base; calibrator_hidden=1024; semantic_alignment=0.25; semantic_contrastive=1.0; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=0; content_token_ce_weight=2; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=1.0; content_keyword_context=1; content_keyword_train_context=1 ;;
  45) variant=prefix16_oofordered_oracleoofkeyword_exactlex_raw_b1; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=5; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=1.0; content_keyword_context=1; content_keyword_train_context=1; oracle_keyword_train_context=1; content_keyword_template=raw ;;
  46) variant=prefix16_oofordered_oracleoofkeyword_full20lex_raw_b1; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=5; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0_full20s_to_exact.pt; content_bias_topk=5; content_bias_strength=1.0; content_keyword_context=1; content_keyword_train_context=1; oracle_keyword_train_context=1; content_keyword_template=raw ;;
  47) variant=prefix16_oofordered_oracleoofkeyword_exactlex_summarize_b1; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=5; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=1.0; content_keyword_context=1; content_keyword_train_context=1; oracle_keyword_train_context=1; content_keyword_template=summarize ;;
  48) variant=prefix16_oofordered_oracleoofkeyword_full20lex_summarize_b1; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=5; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0_full20s_to_exact.pt; content_bias_topk=5; content_bias_strength=1.0; content_keyword_context=1; content_keyword_train_context=1; oracle_keyword_train_context=1; content_keyword_template=summarize ;;
  49) variant=prefix16_oofordered_full20lex_contentce2_pair0p1_keyword5train_lexb1_lora4; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=10; content_token_ce_weight=2; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0_full20s_to_exact.pt; content_bias_topk=5; content_bias_strength=1.5; content_bias_once=1; content_keyword_context=1; content_keyword_train_context=1; t5_lora_rank=4 ;;
  50) variant=prefix16_oofordered_full20lex_contentce2_pair0p1_keyword5train_lexb1_lora8; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=10; content_token_ce_weight=2; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0_full20s_to_exact.pt; content_bias_topk=5; content_bias_strength=1.5; content_bias_once=1; content_keyword_context=1; content_keyword_train_context=1; t5_lora_rank=8; t5_lora_alpha=16 ;;
  51) variant=prefix16_oofordered_contentce2_pair0p1_exactlex_b2_lora4; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=10; content_token_ce_weight=2; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=2.0; content_bias_once=1; t5_lora_rank=4 ;;
  52) variant=prefix16_oofordered_full20lex_contentce2_pair0p1_keyword5train_jointprojector; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; brain_unfreeze_projector=1; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=10; content_token_ce_weight=2; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0_full20s_to_exact.pt; content_bias_topk=5; content_bias_strength=1.5; content_bias_once=1; content_keyword_context=1; content_keyword_train_context=1 ;;
  53) variant=prefix16_oofordered_full20lex_contentce2_pair0p1_keyword5train_jointprojector_lora4; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; brain_unfreeze_projector=1; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=10; content_token_ce_weight=2; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0_full20s_to_exact.pt; content_bias_topk=5; content_bias_strength=1.5; content_bias_once=1; content_keyword_context=1; content_keyword_train_context=1; t5_lora_rank=4 ;;
  54) variant=prefix16_oofordered_full20lex_slowwarm1e4_brain1e5; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=1e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=30; semantic_warmup_lr=1e-4; content_token_ce_weight=2; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0_full20s_to_exact.pt; content_bias_topk=5; content_bias_strength=1.5; content_bias_once=1; content_keyword_context=1; content_keyword_train_context=1 ;;
  55) variant=prefix16_oofordered_full20lex_slowwarm3e5_brain1e5; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=1e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=30; semantic_warmup_lr=3e-5; content_token_ce_weight=2; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0_full20s_to_exact.pt; content_bias_topk=5; content_bias_strength=1.5; content_bias_once=1; content_keyword_context=1; content_keyword_train_context=1 ;;
  56) variant=prefix16_oofordered_full20lex_slowwarm1e4_alignonly_brain1e5; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0; brain_lr=1e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=30; semantic_warmup_lr=1e-4; content_token_ce_weight=2; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0_full20s_to_exact.pt; content_bias_topk=5; content_bias_strength=1.5; content_bias_once=1; content_keyword_context=1; content_keyword_train_context=1 ;;
  57) variant=prefix16_oofordered_idfce2p0p5_pair0p1_exactlex; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=10; content_token_ce_weight=2; content_idf_power=0.5; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=2; content_bias_once=1 ;;
  58) variant=prefix16_oofordered_idfce2p1_pair0p1_exactlex; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=10; content_token_ce_weight=2; content_idf_power=1; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=2; content_bias_once=1 ;;
  59) variant=prefix16_oofordered_idfce2p0p5_pair0p1_full20lex; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=10; content_token_ce_weight=2; content_idf_power=0.5; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0_full20s_to_exact.pt; content_bias_topk=5; content_bias_strength=2; content_bias_once=1 ;;
  60) variant=prefix16_full20oracle_to_exact10_idfce2p0p5_exactlex; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oracle_semantic_npz=$data_root/minilm/temporally_aligned_targets_20260831/full20s/uts01_minilm_l6v2_full20s_train_val_minilm384.npz; oof_delay_mode=ordered; semantic_warmup_epochs=10; content_token_ce_weight=2; content_idf_power=0.5; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=2; content_bias_once=1 ;;
  61) variant=prefix16_full20oracle_to_exact10_idfce2p0p5_full20lex; calibrator_hidden=1024; semantic_alignment=1; semantic_contrastive=0.1; brain_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oracle_semantic_npz=$data_root/minilm/temporally_aligned_targets_20260831/full20s/uts01_minilm_l6v2_full20s_train_val_minilm384.npz; oof_delay_mode=ordered; semantic_warmup_epochs=10; content_token_ce_weight=2; content_idf_power=0.5; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0_full20s_to_exact.pt; content_bias_topk=5; content_bias_strength=2; content_bias_once=1 ;;
  62) variant=prefix16_delayweighted_exact_lr3e4_alignonly; brain_calibrator_kind=delay_weighted; semantic_alignment=1; semantic_contrastive=0; brain_lr=3e-4; semantic_warmup_lr=3e-4; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=50; content_token_ce_weight=2; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=2; content_bias_once=1 ;;
  63) variant=prefix16_delayweighted_exact_lr1e4_alignonly; brain_calibrator_kind=delay_weighted; semantic_alignment=1; semantic_contrastive=0; brain_lr=1e-4; semantic_warmup_lr=1e-4; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=50; content_token_ce_weight=2; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=2; content_bias_once=1 ;;
  64) variant=prefix16_delayweighted_exact_lr3e5_alignonly; brain_calibrator_kind=delay_weighted; semantic_alignment=1; semantic_contrastive=0; brain_lr=3e-5; semantic_warmup_lr=3e-5; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oof_delay_mode=ordered; semantic_warmup_epochs=50; content_token_ce_weight=2; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=2; content_bias_once=1 ;;
  65) variant=prefix16_delayweighted_full20oracle_lr1e4_full20lex; brain_calibrator_kind=delay_weighted; semantic_alignment=1; semantic_contrastive=0; brain_lr=1e-4; semantic_warmup_lr=1e-4; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oracle_semantic_npz=$data_root/minilm/temporally_aligned_targets_20260831/full20s/uts01_minilm_l6v2_full20s_train_val_minilm384.npz; oof_delay_mode=ordered; semantic_warmup_epochs=50; content_token_ce_weight=2; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0_full20s_to_exact.pt; content_bias_topk=5; content_bias_strength=2; content_bias_once=1 ;;
  66) variant=prefix16_delayweighted_full20oracle_lr1e4_exactlex; brain_calibrator_kind=delay_weighted; semantic_alignment=1; semantic_contrastive=0; brain_lr=1e-4; semantic_warmup_lr=1e-4; oof_semantic=$data_root/minilm/oof_robustness/uts01_minilm_delayed1536_middle10content_oracle_plus_ooftrain_predval_v1.npz; oracle_semantic_npz=$data_root/minilm/temporally_aligned_targets_20260831/full20s/uts01_minilm_l6v2_full20s_train_val_minilm384.npz; oof_delay_mode=ordered; semantic_warmup_epochs=50; content_token_ce_weight=2; pairing_margin_weight=0.1; content_head=/data/engs-pnpl/glandau/elf-runs/fmri_oof_content_probe_20260831/oofmean_linear_pair1p0.pt; content_bias_topk=5; content_bias_strength=2; content_bias_once=1 ;;
  *) echo "Unsupported task $task" >&2; exit 2 ;;
esac
variant=${variant}${RUN_SUFFIX:-}
if [[ -z "${RUN_SUFFIX:-}" ]] && \
   [[ -n "${ORACLE_EPOCHS_OVERRIDE:-}" || -n "${BRAIN_EPOCHS_OVERRIDE:-}" ]]; then
  variant=${variant}_shortprobe
fi

output=/data/engs-pnpl/glandau/elf-runs/fmri_minilm_t5_prefix_20260831/${variant}
oof_args=()
if [[ -n "$oof_semantic" ]]; then
  oof_args=(--oof-semantic-npz "$oof_semantic")
fi
oracle_semantic_args=()
if [[ -n "$oracle_semantic_npz" ]]; then
  oracle_semantic_args=(--oracle-semantic-npz "$oracle_semantic_npz")
fi
content_args=()
if [[ -n "$content_head" ]]; then
  content_args=(
    --content-head-checkpoint "$content_head"
    --content-bias-topk "$content_bias_topk"
    --content-bias-strength "$content_bias_strength"
    --content-prior-subtraction 0.5
  )
  if [[ "$content_keyword_context" == 1 ]]; then
    content_args+=(
      --content-keyword-context
      --content-keyword-strength "$content_keyword_strength"
    )
  fi
fi
content_bias_once_args=()
if [[ "$content_bias_once" == 1 ]]; then
  content_bias_once_args=(--content-bias-once)
fi
keyword_train_args=()
if [[ "$content_keyword_train_context" == 1 ]]; then
  keyword_train_args=(--content-keyword-train-context)
fi
oracle_keyword_train_args=()
if [[ "$oracle_keyword_train_context" == 1 ]]; then
  oracle_keyword_train_args=(--oracle-keyword-train-context)
fi
brain_unfreeze_projector_args=()
if [[ "$brain_unfreeze_projector" == 1 ]]; then
  brain_unfreeze_projector_args=(--brain-unfreeze-projector)
fi
cd "$project_root"
export PYTHONPATH="$project_root:$project_root/src:$project_root/scripts:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$data_root/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
"$python_bin" scripts/train_fmri_minilm_t5_prefix.py \
  --brain-npz "$mri_run/dataset_all_minilm_l6_v2_tr2s_delayed_d1-2-3-4_seg10.npz" \
  --text-npz "$data_root/minilm/knowntext_unseenbrain_v1/oracle_exact_minilm384_train_val.npz" \
  --mri-checkpoint "$checkpoint" \
  --output-dir "$output" \
  --model-name "$model_name" \
  --prefix-length "$prefix_length" \
  --projector-hidden-dim "$hidden" \
  --projector-kind "$projector_kind" \
  --brain-calibrator-hidden-dim "$calibrator_hidden" \
  --brain-calibrator-kind "$brain_calibrator_kind" \
  --brain-semantic-alignment-weight "$semantic_alignment" \
  --brain-semantic-contrastive-weight "$semantic_contrastive" \
  --brain-lr "$brain_lr" \
  --brain-projector-lr "$brain_projector_lr" \
  --brain-semantic-warmup-epochs "${SEMANTIC_WARMUP_EPOCHS_OVERRIDE:-$semantic_warmup_epochs}" \
  --brain-semantic-warmup-lr "$semantic_warmup_lr" \
  --t5-lora-rank "$t5_lora_rank" \
  --t5-lora-alpha "$t5_lora_alpha" \
  --t5-lora-last-n-blocks 2 \
  --t5-lora-lr 1e-5 \
  --content-token-ce-weight "$content_token_ce_weight" \
  --content-idf-power "$content_idf_power" \
  --content-idf-max-multiplier "$content_idf_max_multiplier" \
  --brain-pairing-margin-weight "$pairing_margin_weight" \
  --brain-pairing-margin "$pairing_margin" \
  --content-keyword-template "$content_keyword_template" \
  --oof-delay-mode "$oof_delay_mode" \
  --oracle-epochs "${ORACLE_EPOCHS_OVERRIDE:-20}" \
  --brain-epochs "${BRAIN_EPOCHS_OVERRIDE:-30}" \
  --eval-every "${EVAL_EVERY_OVERRIDE:-5}" \
  --batch-size 64 \
  --num-beams 2 \
  --max-output-words 10 \
  --derangements 3 \
  --seed 49 \
  "${oof_args[@]}" \
  "${oracle_semantic_args[@]}" \
  "${content_args[@]}" \
  "${content_bias_once_args[@]}" \
  "${keyword_train_args[@]}" \
  "${oracle_keyword_train_args[@]}" \
  "${brain_unfreeze_projector_args[@]}"
