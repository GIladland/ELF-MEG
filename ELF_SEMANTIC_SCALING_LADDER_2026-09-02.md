# ELF semantic-corpus scaling ladder

Each scale uses the same validation rows and candidate pool. Exact text/semantic pairs are a deliberately
known-text oracle; validation brain vectors remain evaluation-only. MRI test107 and MEG test26 stay sealed.

## mri_minilm

### Training corpus

| Scale | Unique exact | Unique augmented | Unique total | Physical rows | Exact duplicates in augmentation |
|---|---:|---:|---:|---:|---:|
| x1 | 12098 | 0 | 12098 | 12098 | 0 |
| x4 | 12098 | 36294 | 48392 | 84952 | 0 |
| x8 | 12098 | 84686 | 96784 | 181736 | 0 |

### Fixed-row ladder

| Scale | Rung | Word F1 | Content F1 | WER | BERT F1 | Gen Top-1 | Gen Top-5 | Mean / median rank | Sem Top-1 / Top-5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x1 | exact | 0.7910 | 0.7958 | 0.2658 | 0.6406 | 0.9962 | 0.9962 | 1.14 / 1.0 | — / — |
| x1 | raw | 0.0646 | 0.0089 | 1.0060 | -0.0760 | 0.0188 | 0.0940 | 94.55 / 77.0 | 0.0338 / 0.1278 |
| x1 | adapter | 0.0628 | 0.0183 | 1.0049 | -0.0745 | 0.0150 | 0.0602 | 105.34 / 97.0 | 0.0414 / 0.1090 |
| x4 | exact | 0.7408 | 0.7659 | 0.3305 | 0.5786 | 0.9774 | 0.9962 | 1.15 / 1.0 | — / — |
| x4 | raw | 0.0586 | 0.0086 | 1.0568 | -0.1162 | 0.0150 | 0.0639 | 95.42 / 82.0 | 0.0338 / 0.1278 |
| x4 | adapter | 0.0538 | 0.0155 | 1.0669 | -0.1168 | 0.0226 | 0.0451 | 99.91 / 86.0 | 0.0489 / 0.1353 |
| x8 | exact | 0.7215 | 0.7350 | 0.3477 | 0.5478 | 0.9624 | 0.9887 | 1.12 / 1.0 | — / — |
| x8 | raw | 0.0564 | 0.0096 | 1.0586 | -0.1234 | 0.0150 | 0.0564 | 95.87 / 87.0 | 0.0338 / 0.1278 |
| x8 | adapter | 0.0496 | 0.0159 | 1.0820 | -0.1196 | 0.0150 | 0.0526 | 104.22 / 93.0 | 0.0414 / 0.1241 |

### Information retained from the exact-semantic ceiling

Ratios compare each brain-conditioned rung with the exact ELF at the same corpus scale.

| Scale | Rung | Word-F1 retained | Content-F1 retained | Gen Top-1 retained | Gen Top-5 retained |
|---|---|---:|---:|---:|---:|
| x1 | raw | 0.0816 | 0.0111 | 0.0189 | 0.0943 |
| x1 | adapter | 0.0794 | 0.0230 | 0.0151 | 0.0604 |
| x4 | raw | 0.0791 | 0.0112 | 0.0154 | 0.0642 |
| x4 | adapter | 0.0727 | 0.0202 | 0.0231 | 0.0453 |
| x8 | raw | 0.0782 | 0.0130 | 0.0156 | 0.0570 |
| x8 | adapter | 0.0687 | 0.0216 | 0.0156 | 0.0532 |

### ELF-stage convergence audit

`continue` means the best checkpoint is in the last three evaluations and WER improved by at least 0.003 across that tail.

| Scale | Evaluations | First WER | Best WER @ step | Last WER @ step | Tail gain | Decision |
|---|---:|---:|---:|---:|---:|---|
| x4 | 11 | 0.4372 | 0.3305 @ 9000 | 0.3376 @ 10624 | -0.0071 | plateau |
| x8 | 12 | 0.4726 | 0.3477 @ 10000 | 0.3508 @ 11360 | -0.0030 | plateau |

Best adapted scale by content F1: `x1` (`elf_semantic_scaling_mri_minilm_x1_trained_adapter_val266_seed49_20260902`).

| Row | Content F1 | Target | Prediction |
|---:|---:|---|---|
| 137 | 0.333 | television set and i find the new york mets now | like stretches that happened in a new york |
| 210 | 0.308 | this nonstop loop of my stepfather almost like that cliche | all of things but like almost one bullets with four nuances |
| 133 | 0.286 | stay i left baghdad early the next morning going back | y know i had to get back the next year |
| 54 | 0.200 | would go and continued toward los angeles i began wondering | you can can go the well my dad would admit |
| 229 | 0.200 | college loans and you know bought me an apartment paid | walked and knocks a one of the i know |
| 76 | 0.182 | wrong way to make art now this makes me extremely | physicsman he falls out his way he knows |
| 35 | 0.182 | called us a week or two later this great grandmother | two dozen um well a time i backed up |
| 191 | 0.182 | item you never know what you're modeling for your kids | not know my mileage of typical behavioral stampes |
| 220 | 0.182 | kind of semi lame at my side and you know | know go around with mar's that not too confident threshold |
| 2 | 0.167 | adopt one day and you know how everybody always says | one classroom in law i already had a doc |

## meg_ada

### Training corpus

| Scale | Unique exact | Unique augmented | Unique total | Physical rows | Exact duplicates in augmentation |
|---|---:|---:|---:|---:|---:|
| x1 | 2788 | 0 | 2788 | 2788 | 0 |
| x4 | 2788 | 8364 | 11152 | 19626 | 0 |
| x16 | 2788 | 41820 | 44608 | 86538 | 0 |

### Fixed-row ladder

| Scale | Rung | Word F1 | Content F1 | WER | BERT F1 | Gen Top-1 | Gen Top-5 | Mean / median rank | Sem Top-1 / Top-5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x1 | exact | 0.8815 | 0.8917 | 0.1455 | 0.7814 | 0.9909 | 1.0000 | 1.01 / 1.0 | — / — |
| x1 | raw | 0.0454 | 0.0155 | 1.0455 | -0.0803 | 0.0455 | 0.1182 | 41.90 / 37.0 | 0.2909 / 0.4545 |
| x1 | adapter | 0.0620 | 0.0298 | 1.0036 | -0.0820 | 0.0364 | 0.1364 | 40.14 / 28.0 | 0.2909 / 0.4545 |
| x4 | exact | 0.8692 | 0.8622 | 0.1582 | 0.7846 | 1.0000 | 1.0000 | 1.00 / 1.0 | — / — |
| x4 | raw | 0.0402 | 0.0132 | 1.0500 | -0.1013 | 0.0364 | 0.1364 | 39.55 / 30.0 | 0.2909 / 0.4545 |
| x4 | adapter | 0.0612 | 0.0269 | 1.0455 | -0.0791 | 0.0091 | 0.1091 | 40.47 / 32.0 | 0.2909 / 0.4545 |
| x16 | exact | 0.8846 | 0.8876 | 0.1409 | 0.8020 | 1.0000 | 1.0000 | 1.00 / 1.0 | — / — |
| x16 | raw | 0.0421 | 0.0194 | 1.0664 | -0.0969 | 0.0364 | 0.1091 | 41.94 / 33.0 | 0.2909 / 0.4545 |
| x16 | adapter | 0.0597 | 0.0272 | 1.0391 | -0.0862 | 0.0364 | 0.1364 | 40.26 / 38.0 | 0.2909 / 0.4545 |

### Information retained from the exact-semantic ceiling

Ratios compare each brain-conditioned rung with the exact ELF at the same corpus scale.

| Scale | Rung | Word-F1 retained | Content-F1 retained | Gen Top-1 retained | Gen Top-5 retained |
|---|---|---:|---:|---:|---:|
| x1 | raw | 0.0515 | 0.0174 | 0.0459 | 0.1182 |
| x1 | adapter | 0.0703 | 0.0334 | 0.0367 | 0.1364 |
| x4 | raw | 0.0462 | 0.0154 | 0.0364 | 0.1364 |
| x4 | adapter | 0.0704 | 0.0312 | 0.0091 | 0.1091 |
| x16 | raw | 0.0475 | 0.0219 | 0.0364 | 0.1091 |
| x16 | adapter | 0.0675 | 0.0306 | 0.0364 | 0.1364 |

### ELF-stage convergence audit

`continue` means the best checkpoint is in the last three evaluations and WER improved by at least 0.003 across that tail.

| Scale | Evaluations | First WER | Best WER @ step | Last WER @ step | Tail gain | Decision |
|---|---:|---:|---:|---:|---:|---|
| x4 | 13 | 0.2409 | 0.1582 @ 4000 | 0.1636 @ 12280 | 0.0055 | plateau |
| x16 | 13 | 0.2455 | 0.1409 @ 11000 | 0.1491 @ 12177 | -0.0082 | plateau |

Best adapted scale by content F1: `x1` (`elf_semantic_scaling_meg_ada_x1_trained_adapter_val110_seed49_20260902`).

| Row | Content F1 | Target | Prediction |
|---:|---:|---|---|
| 72 | 0.222 | us military you will forever be an al qaeda target | nation to back the ehlateral the military |
| 15 | 0.222 | eyes closed and her mouth open and blood was trickling | dog was the  underneath a blood and it |
| 62 | 0.200 | the street he turned left and instead of turning around | go home and do instead of he and the stop |
| 18 | 0.182 | this nonstop loop of my stepfather almost like that cliche | bobppy  start whiping up into the hill like |
| 20 | 0.182 | kept getting louder and louder the silence then i finally | started getting hurting back to my whiy |
| 59 | 0.182 | i wanted i asked to go on vacation to see | go home avan a year up now wests |
| 26 | 0.167 | took us a long time to heal my stepfather broke | the home detached and she took to home her wedding job |
| 27 | 0.167 | one nursing twenty four hours around the clock at home | a front side brunite's home by |
| 31 | 0.167 | multiple surgeries on the arm every time the arm got | oil piecemate a time to take of of drink |
| 32 | 0.167 | wear a short sleeve blouse or a long sleeve blouse | long time get up but  i started trying to |

