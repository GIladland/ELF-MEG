# MEG → ELF-B nested-OOF gap closure

All systems use the same 110 validation sentences, 110 retrieval candidates, decoding settings, and five generation seeds.

| System | Word F1 | Content F1 | Sum | WER ↓ | Generated T1 | Generated T5 | Mean rank ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `exact_ada` | 0.8889 | 0.8945 | 1.7834 | 0.1365 | 1.0000 | 1.0000 | 1.00 |
| `raw_meg` | 0.0460 | 0.0174 | 0.0634 | 1.0415 | 0.0455 | 0.1418 | 42.99 |
| `oof_flat` | 0.0498 | 0.0201 | 0.0699 | 0.9982 | 0.0418 | 0.1127 | 42.35 |
| `oof_residual` | 0.0437 | 0.0093 | 0.0530 | 1.0058 | 0.0291 | 0.1273 | 41.90 |

## Fraction of the raw-MEG → exact-ADA gap closed

`0` means no gain over raw MEG; `1` reaches the exact-ADA oracle. Negative values regress.

| Adapter | Word F1 | Content F1 | Sum | WER | Generated T1 | Generated T5 | Mean rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| `oof_flat` | 0.0045 | 0.0030 | 0.0038 | 0.0478 | -0.0038 | -0.0339 | 0.0150 |
| `oof_residual` | -0.0028 | -0.0093 | -0.0061 | 0.0394 | -0.0171 | -0.0169 | 0.0259 |

## OOF audit

- OOF coverage: no duplicates for 2652 rows; partial=False; outer predictions were not used for checkpoint selection.
- OOF train retrieval (2,652-way): Top-1 `0.0128`, Top-5 `0.0468`, cosine `0.6598`.
- Deployment validation retrieval (110-way): Top-1 `0.2909`, Top-5 `0.4545`, cosine `0.7174`.
- The candidate counts differ in the diagnostic above; the generation table itself is strictly 110-way for every system.
