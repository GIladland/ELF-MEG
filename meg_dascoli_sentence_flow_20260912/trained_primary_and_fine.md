# Confidence-weighted D'Ascoli sentence-source flow results

Target-derived D'Ascoli simulations on the established 110-row validation tail. These are sensitivity/ceiling experiments, not measured brain-decoding results. Protected test26 is absent.

| Run | Mode | Sem. scale | Conf. floor | Nominal acc. | Paired source acc. | Direct source sum | Output content F1 | Output word F1 | Sum | Δ sum vs semantic-only | WER ↓ | Top-1 | Top-5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_joint_shuffled_acc0.50_seed49_20260911 | joint_shuffled |  |  | 0.50 | 0.00636 | 0.03849 | 0.02173 | 0.06593 | 0.08766 | 0.01018 | 1.01818 | 0.04545 | 0.16364 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_joint_unweighted_acc0.50_seed49_20260911 | joint_unweighted |  |  | 0.50 | 0.49818 | 0.95226 | 0.05895 | 0.13960 | 0.19854 | 0.12106 | 0.93636 | 0.15455 | 0.34545 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_joint_weighted_acc0.50_seed49_20260911 | joint_weighted |  |  | 0.50 | 0.49818 | 0.95226 | 0.02882 | 0.09568 | 0.12450 | 0.04702 | 0.97636 | 0.11818 | 0.30909 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_sem0001_floor1_joint_weighted_acc0.50_seed49_20260911 | joint_weighted | 0.001 | 1.00 | 0.50 | 0.49818 | 0.95226 | 0.28264 | 0.35989 | 0.64254 | 0.56505 | 0.70000 | 0.42727 | 0.66364 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_sem001_floor1_joint_weighted_acc0.50_seed49_20260911 | joint_weighted | 0.010 | 1.00 | 0.50 | 0.49818 | 0.95226 | 0.28371 | 0.36098 | 0.64468 | 0.56720 | 0.70000 | 0.41818 | 0.65455 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_sem0025_floor1_joint_weighted_acc0.50_seed49_20260911 | joint_weighted | 0.025 | 1.00 | 0.50 | 0.49818 | 0.95226 | 0.28801 | 0.36439 | 0.65240 | 0.57492 | 0.68273 | 0.44545 | 0.65455 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_sem002_floor1_joint_weighted_acc0.50_seed49_20260911 | joint_weighted | 0.020 | 1.00 | 0.50 | 0.49818 | 0.95226 | 0.28671 | 0.36463 | 0.65134 | 0.57385 | 0.68455 | 0.43636 | 0.67273 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_sem0035_floor1_joint_weighted_acc0.50_seed49_20260911 | joint_weighted | 0.035 | 1.00 | 0.50 | 0.49818 | 0.95226 | 0.28839 | 0.36658 | 0.65497 | 0.57749 | 0.66727 | 0.47273 | 0.69091 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_sem003_floor1_joint_weighted_acc0.50_seed49_20260911 | joint_weighted | 0.030 | 1.00 | 0.50 | 0.49818 | 0.95226 | 0.28853 | 0.36471 | 0.65325 | 0.57576 | 0.68000 | 0.46364 | 0.66364 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_sem004_floor1_joint_weighted_acc0.50_seed49_20260911 | joint_weighted | 0.040 | 1.00 | 0.50 | 0.49818 | 0.95226 | 0.28367 | 0.36204 | 0.64570 | 0.56822 | 0.67182 | 0.48182 | 0.67273 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_sem005_floor1_joint_weighted_acc0.50_seed49_20260911 | joint_weighted | 0.050 | 1.00 | 0.50 | 0.49818 | 0.95226 | 0.27209 | 0.35461 | 0.62670 | 0.54921 | 0.67909 | 0.45455 | 0.65455 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_sem0075_floor1_joint_weighted_acc0.50_seed49_20260911 | joint_weighted | 0.075 | 1.00 | 0.50 | 0.49818 | 0.95226 | 0.21889 | 0.32209 | 0.54098 | 0.46350 | 0.71000 | 0.40909 | 0.59091 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_semantic_only_acc0.50_seed49_20260911 | semantic_only |  |  | 0.50 | 0.49818 | 0.95226 | 0.01763 | 0.05985 | 0.07748 | 0.00000 | 1.00545 | 0.06364 | 0.14545 |
| qc4wyals_elfb_dascoli_eval_lora8_mid_ep40_sentence_only_acc0.50_seed49_20260911 | sentence_only |  |  | 0.50 | 0.49818 | 0.95226 | 0.22234 | 0.29909 | 0.52143 | 0.44395 | 0.87364 | 0.43636 | 0.61818 |
