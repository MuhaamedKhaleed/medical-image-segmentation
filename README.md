# Paired-Mask U-Net Research

Strict local image/mask validation and explicit CPU U-Net training. Missing masks fail instead of silently becoming all-zero labels (a bug in the original notebook that could make an invalid experiment appear to train). **Not a diagnostic device or a validated clinical model.**

## Results (honest)

- Fresh-clone CPU run on CC0 BBBC039 microscopy nuclei (25 Nov 2026, ~25 s training on one core): **test Dice 0.8964, development Dice 0.8928** on 16 official test fields. This is nuclei microscopy, **not the original brain MRI** - those weights were never recovered.
- The test fields were observed in earlier runs, so treat this as a reproducible demonstration, not an untouched benchmark. Training is not bit-deterministic; expect small differences.
- The 2-epoch `real_data.py` route is only a wiring check (Dice ~0.07).
- 24 tests, 96% statement coverage (tiny-width U-Net forward/train/reload, mask/dimension/split guards). Coverage is not medical accuracy.

## Quickstart (fresh clone, about 1 minute)

Needs Python 3.11 or 3.12 (locked wheels have no 3.13+ build), internet for the first install/downloads, ~2 GB free RAM.

```sh
uv sync --python 3.11 --locked --extra dev --extra ml
uv run --no-sync python scripts/fetch_real_data.py --output ./real-source          # ~80 MB, CC0 BBBC039 microscopy
uv run --no-sync python scripts/quality_cpu.py --data ./real-source --output ./seg-out
```

Output: `seg-out/best.keras` (trained U-Net), `report.json` (Dice) and `test-pred.png`. Your own data needs a manifest CSV with `image,mask,patient` headers; patient IDs drive the seeded split. `medical-segmentation check-data --data YOUR.csv --size 32` validates it.

## Limits

- Validation Dice pools pixels across the batch (not mean per-patient), and empty-mask prevalence can distort it. No clinical sensitivity/specificity, calibration or subgroup quality is claimed.
- Patient-disjoint splitting is only as correct as the supplied IDs; duplicate image content at different paths is not detected. Consent, deidentification and licensing review are required before using real medical data.
- Default full-size training (input 128, base filters 64) preserves the original U-Net and can consume substantial RAM/time; demo runs reduce it explicitly.
- Verified on Linux x86_64; the container runs data checks only (no TensorFlow). Free hosted CI covers tests, lint, CLI and Docker build.
- No patient data, models or datasets are published; no release or deployment.

## License

MIT for the new project-owned code (see `LICENSE` and `LICENSE-SCOPE.md`). Datasets and historical code keep their own terms. Details and evidence: `docs/verification.md`, `docs/real-data.md`, `docs/quality-cpu.md`, `CHANGELOG.md`.
