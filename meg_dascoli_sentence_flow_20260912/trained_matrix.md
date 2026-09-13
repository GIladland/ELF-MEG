# Confidence-weighted D'Ascoli sentence-source flow results

Target-derived D'Ascoli simulations on the established 110-row validation tail. These are sensitivity/ceiling experiments, not measured brain-decoding results. Protected test26 is absent.

| Run | Mode | Nominal acc. | Paired source acc. | Direct source sum | Output content F1 | Output word F1 | Sum | Δ sum vs semantic-only | WER ↓ | Top-1 | Top-5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_joint_shuffled_acc0.50_seed49_20260911 | joint_shuffled | 0.50 | 0.00636 | 0.03849 | 0.02173 | 0.06593 | 0.08766 | 0.01018 | 1.01818 | 0.04545 | 0.16364 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_joint_unweighted_acc0.50_seed49_20260911 | joint_unweighted | 0.50 | 0.49818 | 0.95226 | 0.05895 | 0.13960 | 0.19854 | 0.12106 | 0.93636 | 0.15455 | 0.34545 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_joint_weighted_acc0.00_seed49_20260911 | joint_weighted | 0.00 | 0.00000 | 0.01020 | 0.01967 | 0.05901 | 0.07868 | 0.00119 | 1.00636 | 0.02727 | 0.15455 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_joint_weighted_acc0.25_seed49_20260911 | joint_weighted | 0.25 | 0.23091 | 0.44588 | 0.01964 | 0.06249 | 0.08212 | 0.00464 | 0.99273 | 0.04545 | 0.16364 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_joint_weighted_acc0.50_seed49_20260911 | joint_weighted | 0.50 | 0.49818 | 0.95226 | 0.02882 | 0.09568 | 0.12450 | 0.04702 | 0.97636 | 0.11818 | 0.30909 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_joint_weighted_acc0.75_seed49_20260911 | joint_weighted | 0.75 | 0.75545 | 1.48318 | 0.04953 | 0.13377 | 0.18330 | 0.10582 | 0.93545 | 0.18182 | 0.37273 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_joint_weighted_acc0.90_seed49_20260911 | joint_weighted | 0.90 | 0.89727 | 1.78017 | 0.05851 | 0.14188 | 0.20039 | 0.12291 | 0.91364 | 0.21818 | 0.42727 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_joint_weighted_acc1.00_seed49_20260911 | joint_weighted | 1.00 | 1.00000 | 2.00000 | 0.06454 | 0.15436 | 0.21890 | 0.14142 | 0.90818 | 0.28182 | 0.48182 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_semantic_only_acc0.50_seed49_20260911 | semantic_only | 0.50 | 0.49818 | 0.95226 | 0.01763 | 0.05985 | 0.07748 | 0.00000 | 1.00545 | 0.06364 | 0.14545 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_sentence_only_acc0.50_seed49_20260911 | sentence_only | 0.50 | 0.49818 | 0.95226 | 0.22234 | 0.29909 | 0.52143 | 0.44395 | 0.87364 | 0.43636 | 0.61818 |
