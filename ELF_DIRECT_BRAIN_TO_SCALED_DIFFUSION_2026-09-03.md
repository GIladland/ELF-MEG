# Direct brain vectors into scaled ELF diffusion

Only the ELF semantic-to-text training-corpus scale changes. Brain-predicted vectors, validation rows,
candidate pools, checkpoint seed, and diffusion generation settings are fixed within each modality.
MRI uses the same deterministic untrained delayed-vector input reduction at every scale; MEG passes
the 1536-dimensional predicted ADA vector directly through the checkpoint's flat projection.

## MRI / MiniLM: fixed 266-way evaluation

| ELF scale | Unique pairs | Exact content F1 | Exact WER | Direct word F1 | Direct content F1 | Direct WER | BERT F1 | Gen Top-1 | Gen Top-5 | Mean / median rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x1 | 12098 | 0.7958 | 0.2658 | 0.0646 | 0.0089 | 1.0060 | -0.0760 | 0.0188 | 0.0940 | 94.55 / 77.0 |
| x4 | 48392 | 0.7659 | 0.3305 | 0.0586 | 0.0086 | 1.0568 | -0.1162 | 0.0150 | 0.0639 | 95.42 / 82.0 |
| x8 | 96784 | — | — | — | — | — | — | — | — | — / — |

| ELF scale | Content retained vs exact | Δ content F1 vs x1 | Δ WER vs x1 | Δ BERT vs x1 | Δ Top-1 vs x1 | Δ Top-5 vs x1 | Brain semantic Top-1 / Top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| x1 | 0.0111 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0338 / 0.1278 |
| x4 | 0.0112 | -0.0003 | 0.0508 | -0.0402 | -0.0038 | -0.0301 | 0.0338 / 0.1278 |
| x8 | — | — | — | — | — | — | — / — |

Paired x1 -> x4 deltas (20,000 bootstrap samples):

| Metric | Delta | 95% CI | Probability larger ELF is better |
|---|---:|---:|---:|
| Content F1 | -0.0003 | [-0.0065, +0.0056] | 0.4667 |
| Word F1 | -0.0059 | [-0.0146, +0.0028] | 0.0880 |
| Per-row WER | 0.0508 | [+0.0353, +0.0662] | 0.0000 |
| BERTScore F1 | -0.0402 | [-0.0509, -0.0297] | 0.0000 |

## MEG / ADA: fixed 110-way evaluation

| ELF scale | Unique pairs | Exact content F1 | Exact WER | Direct word F1 | Direct content F1 | Direct WER | BERT F1 | Gen Top-1 | Gen Top-5 | Mean / median rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x1 | 2788 | 0.8917 | 0.1455 | 0.0454 | 0.0155 | 1.0455 | -0.0803 | 0.0455 | 0.1182 | 41.90 / 37.0 |
| x4 | 11152 | — | — | — | — | — | — | — | — | — / — |
| x16 | 44608 | 0.8876 | 0.1409 | 0.0421 | 0.0194 | 1.0664 | -0.0969 | 0.0364 | 0.1091 | 41.94 / 33.0 |

| ELF scale | Content retained vs exact | Δ content F1 vs x1 | Δ WER vs x1 | Δ BERT vs x1 | Δ Top-1 vs x1 | Δ Top-5 vs x1 | Brain semantic Top-1 / Top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| x1 | 0.0174 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2909 / 0.4545 |
| x4 | — | — | — | — | — | — | — / — |
| x16 | 0.0219 | 0.0039 | 0.0209 | -0.0166 | -0.0091 | -0.0091 | 0.2909 / 0.4545 |

Paired x1 -> x16 deltas (20,000 bootstrap samples):

| Metric | Delta | 95% CI | Probability larger ELF is better |
|---|---:|---:|---:|
| Content F1 | 0.0039 | [-0.0061, +0.0143] | 0.7714 |
| Word F1 | -0.0034 | [-0.0162, +0.0096] | 0.2985 |
| Per-row WER | 0.0209 | [+0.0027, +0.0400] | 0.0126 |
| BERTScore F1 | -0.0166 | [-0.0290, -0.0039] | 0.0055 |

