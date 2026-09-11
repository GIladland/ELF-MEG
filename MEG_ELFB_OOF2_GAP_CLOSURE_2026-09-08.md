# MEG → ELF-B nested-OOF gap closure

All systems use the same 110 validation sentences, 110 retrieval candidates, decoding settings, and five generation seeds.

| System | Word F1 | Content F1 | Sum | WER ↓ | Generated T1 | Generated T5 | Mean rank ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `exact_ada` | 0.8889 | 0.8945 | 1.7834 | 0.1365 | 1.0000 | 1.0000 | 1.00 |
| `raw_meg` | 0.0460 | 0.0174 | 0.0634 | 1.0415 | 0.0455 | 0.1418 | 42.99 |
| `oof2_flat` | 0.0511 | 0.0220 | 0.0731 | 1.0089 | 0.0309 | 0.1145 | 43.24 |
| `oof2_residual` | 0.0462 | 0.0140 | 0.0603 | 1.0236 | 0.0418 | 0.1091 | 41.88 |

## Fraction of the raw-MEG → exact-ADA gap closed

`0` means no gain over raw MEG; `1` reaches the exact-ADA oracle. Negative values regress.

| Adapter | Word F1 | Content F1 | Sum | WER | Generated T1 | Generated T5 | Mean rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| `oof2_flat` | 0.0060 | 0.0052 | 0.0056 | 0.0360 | -0.0152 | -0.0318 | -0.0060 |
| `oof2_residual` | 0.0003 | -0.0039 | -0.0019 | 0.0197 | -0.0038 | -0.0381 | 0.0264 |

## OOF audit

- OOF coverage: no duplicates for 1079 rows; partial=True; outer predictions were not used for checkpoint selection.
- OOF train retrieval (1,079-way): Top-1 `0.0324`, Top-5 `0.1038`, cosine `0.6240`.
- Deployment validation retrieval (110-way): Top-1 `0.2909`, Top-5 `0.4545`, cosine `0.7174`.
- The candidate counts differ in the diagnostic above; the generation table itself is strictly 110-way for every system.
