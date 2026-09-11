# ELF ARC cleanup — 2026-09-01

- Permanently deleted `810` obsolete top-level entries from
  `/data/engs-pnpl/glandau/elf-runs`.
- Canceled the five remaining obsolete ELF jobs before deletion:
  `8692287`, `8692372`, `8692428`, `8694590`, and `8671824`.
- Preserved the reload-verified MRI-to-ELF diffusion content winner:

  `/data/engs-pnpl/glandau/elf-runs/fmri_minilm_elf_xattn_last2_lr1e4_from_recover2_step1000_20260831_lexgate_top5_frompartstep37_top10_prior0p1_t2_scattered_b40_val266_generationonly/packaged_system/fmri_elf_content_first_val266_20260831.pt`

- Preserved size: `1,893,042,242` bytes.
- Preserved SHA-256:
  `a4031717b2f8b5095998fd45bb922b43d7b3385ce5b9b3aeefd1ea7b56b4fe54`.
- Verified the checksum both before and after cleanup.
- Final `elf-runs` size: approximately `1.8 GB`.
- `/data/engs-pnpl` changed from `100%` full to `92%`, with approximately
  `1.3 TB` available immediately after cleanup.
- `elf-cache`, datasets, Conda environments, Hugging Face/model caches, source
  code, and local reports were not deleted.

The deleted run artifacts were removed permanently and are not recoverable from
trash. They can be regenerated from source and retained experiment reports.
