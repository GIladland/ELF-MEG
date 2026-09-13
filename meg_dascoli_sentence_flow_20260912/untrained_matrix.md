# Confidence-weighted D'Ascoli sentence-source flow results

Target-derived D'Ascoli simulations on the established 110-row validation tail. These are sensitivity/ceiling experiments, not measured brain-decoding results. Protected test26 is absent.

| Run | Mode | Sem. scale | Conf. floor | Nominal acc. | Paired source acc. | Direct source sum | Output content F1 | Output word F1 | Sum | Δ sum vs semantic-only | WER ↓ | Top-1 | Top-5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qc4wyals_elfb_dascoli_eval_x1_untrained_joint_shuffled_acc0.50_seed49_20260911 | joint_shuffled |  |  | 0.50 | 0.00636 | 0.03849 | 0.02147 | 0.05298 | 0.07445 | -0.00531 | 1.01273 | 0.01818 | 0.15455 |
| qc4wyals_elfb_dascoli_eval_x1_untrained_joint_unweighted_acc0.50_seed49_20260911 | joint_unweighted |  |  | 0.50 | 0.49818 | 0.95226 | 0.04773 | 0.12242 | 0.17015 | 0.09038 | 0.94091 | 0.21818 | 0.34545 |
| qc4wyals_elfb_dascoli_eval_x1_untrained_joint_weighted_acc0.00_seed49_20260911 | joint_weighted |  |  | 0.00 | 0.00000 | 0.01020 | 0.01913 | 0.05333 | 0.07246 | -0.00730 | 1.00545 | 0.07273 | 0.15455 |
| qc4wyals_elfb_dascoli_eval_x1_untrained_joint_weighted_acc0.25_seed49_20260911 | joint_weighted |  |  | 0.25 | 0.23091 | 0.44588 | 0.01902 | 0.06184 | 0.08085 | 0.00109 | 1.00727 | 0.06364 | 0.22727 |
| qc4wyals_elfb_dascoli_eval_x1_untrained_joint_weighted_acc0.50_seed49_20260911 | joint_weighted |  |  | 0.50 | 0.49818 | 0.95226 | 0.02487 | 0.07910 | 0.10397 | 0.02421 | 0.98455 | 0.16364 | 0.30000 |
| qc4wyals_elfb_dascoli_eval_x1_untrained_joint_weighted_acc0.75_seed49_20260911 | joint_weighted |  |  | 0.75 | 0.75545 | 1.48318 | 0.04259 | 0.11227 | 0.15485 | 0.07509 | 0.96000 | 0.17273 | 0.33636 |
| qc4wyals_elfb_dascoli_eval_x1_untrained_joint_weighted_acc0.90_seed49_20260911 | joint_weighted |  |  | 0.90 | 0.89727 | 1.78017 | 0.05985 | 0.12192 | 0.18177 | 0.10201 | 0.94727 | 0.23636 | 0.40909 |
| qc4wyals_elfb_dascoli_eval_x1_untrained_joint_weighted_acc1.00_seed49_20260911 | joint_weighted |  |  | 1.00 | 1.00000 | 2.00000 | 0.06692 | 0.14047 | 0.20739 | 0.12762 | 0.92636 | 0.26364 | 0.48182 |
| qc4wyals_elfb_dascoli_eval_x1_untrained_semantic_only_acc0.50_seed49_20260911 | semantic_only |  |  | 0.50 | 0.49818 | 0.95226 | 0.02285 | 0.05691 | 0.07976 | 0.00000 | 1.00909 | 0.04545 | 0.13636 |
| qc4wyals_elfb_dascoli_eval_x1_untrained_sentence_only_acc0.50_seed49_20260911 | sentence_only |  |  | 0.50 | 0.49818 | 0.95226 | 0.20205 | 0.27582 | 0.47787 | 0.39811 | 1.02909 | 0.31818 | 0.50909 |
