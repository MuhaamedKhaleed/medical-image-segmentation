# Paired-Mask U-Net Research

Strict local image/mask checks and explicit CPU U-Net training for a research workflow. Missing masks fail instead of silently becoming all-zero labels. This is **not a diagnostic device**, a validated clinical model or a demonstration of MRI segmentation quality.

## Why this pass matters

The original notebook downloaded a brain-tumor classification dataset and assumed a separate mask folder. Its loader invented zero masks when files were missing, which could make an invalid experiment appear to train. Classification labels are not pixel annotations. This package requires an explicit manifest of genuine paired masks and patient IDs. No data, checkpoint, API credential or automatic download is bundled.

## Install and demo

Python3.11/3.12, Linux x86_64. With uv installed:

```sh
uv sync --locked --extra dev
uv run medical-segmentation --help
uv run python scripts/synthetic_demo.py
uv run medical-segmentation check-data --data synthetic-demo/manifest.csv --size 32
uv run medical-segmentation check-data --config .env.example --size 32
```

Actual data-check output:

```json
{"pairs": 3, "train_images": 2, "validation_images": 1, "patients": 3}
```

The demo deliberately draws squares, not medical images. It verifies data wiring only. The script refuses to overwrite `synthetic-demo`. `--config` reads only the explicitly selected dotenv file's `MANIFEST_CSV`; `--data` wins. No real secret is required.

## Data contract

UTF-8 CSV headers: `image,mask,patient`. Paths are relative to the manifest. Every row must name an existing image, a distinct existing mask, and a nonempty patient identifier. Repeated image/mask paths, missing fields, different native dimensions, unreadable files and RGB/soft/nonbinary masks fail clearly. Masks must be single-channel values0/1 or0/255. Explicitly supplied empty masks are accepted; they are never fabricated. Image intensities convert to grayscale and scale to [0,1]. Masks resize with nearest-neighbor, images with bilinear.

Patient IDs determine the seeded train/validation split. The program cannot verify those IDs or annotation provenance, and does not detect duplicate image content at different paths. A patient-disjoint split is only as correct as the supplied IDs. It does not establish site/scanner/domain independence. Ensure consent, deidentification, licensing and expert annotation review before handling real data.

## Deliberate local training

```sh
uv sync --locked --extra dev --extra ml
TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 uv run medical-segmentation train synthetic-demo/manifest.csv synthetic-demo/trained --epochs 1 --size 32 --filters 1
```

The command was tested on invented squares only. Output must not exist. It saves `model.keras` and `validation.json` after finite-loss checks. No cloud process starts. Training is CPU-only and explicit. Default input128 and base filters64 preserve the original four-level U-Net widths1024 at the bottleneck; the test/demo uses filters1 for a small wiring test, not an equivalent trained medical model. Large default training can consume substantial RAM/time. There is no API, patient prediction service or automatic artifact upload.

Validation reports binary cross-entropy and global thresholded Dice at0.5, with empty/empty=1. Dice pools pixels across the validation batch, not mean per-patient Dice or lesion sensitivity. This convention and empty-mask prevalence can distort apparent performance. No clinical sensitivity, specificity, real-data Dice, calibration or subgroup quality is claimed.

## Checks

```sh
TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 uv run pytest --cov=medical_segmentation --cov-report=term-missing
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pre-commit run --all-files
uv run python -m medical_segmentation --help
```

24 tests,96% statement coverage. Actual tiny-width U-Net forward/train/save/reload and explicit CLI train run on synthetic data. Meaningful guards cover missing masks, image/mask dimensions, binary masks, patient split and Dice conventions. Light-only install runs data/CLI tests with the TensorFlow test module explicitly skipped. Test coverage is not medical accuracy.

## Docker

```sh
docker build --tag medical-segmentation:ci .
docker run --rm medical-segmentation:ci --help
docker run --rm -v "$PWD/synthetic-demo:/data:ro" medical-segmentation:ci check-data --data /data/manifest.csv --size 32
```

Multi-stage slim, non-root, no ML/dev dependencies: **data checks only**, not a training image. Hosted CI builds/runs these checks; there is no local Docker-daemon claim. CI is public-repository-only on ordinary hosted Linux runners. No paid API, GPU service, deployment or billable private-runner configuration is added. The uv lock is Linuxx86_64-scoped; pip in CI uses direct pins rather than claiming full uv-lock enforcement.

## Structure

```text
manifest CSV -> core.manifest -> pair/mask checks -> patient split
                          -> explicit model.train -> model.keras + metrics
Typer CLI -> separate JSON output / stderr errors
```

`core.py` owns pairs, categorical masks and Dice. `model.py` lazily imports TensorFlow and builds/trains the U-Net. `cli.py` holds thin commands. The notebook is historical, outputs cleared, download/credential cells disabled, missing-mask bug fixed and unverified augmentation removed. **It was not rerun end-to-end.** Use the package for the supported workflow.

Next work: obtain appropriate genuine paired annotations and verified patient IDs, expert review, train/validate/test separation with a frozen external test cohort, lesion-level and per-patient metrics, annotation disagreement and site/scanner shift analysis. No data-independent model-quality improvement is inferred from these engineering fixes.

[Verification](docs/verification.md), [changelog](CHANGELOG.md), [copy-only presentation](docs/presentation.md). No license change, release/tag, main merge or patient/model/data publication.


## License scope

New project-owned package/engineering files use the owner's MIT default. See LICENSE and LICENSE-SCOPE.md. This is not permission to relicense historical notebooks, third-party code, datasets, models or weights. Their provenance and terms still need review. Earlier verification entries saying no license was added describe the initial pilot, not this follow-up.

## Checked synthetic wiring demo

See [commands, saved output and scope](docs/synthetic-demo.md). This is invented-input execution evidence, not model accuracy or a real-data gallery.

## Portable notebook follow-up

The root notebook now uses the supported local package with safe defaults and explicit data/training opt-ins. It replaces the platform-specific experiment rather than reproducing historical results. See `docs/notebook-migration.md` and the notebook settings. Earlier descriptions of live notebook downloads/hard-coded paths are historical and superseded. No dataset or weight rights are granted.

## Fresh-clone path to a trained model (verified 25 Nov 2026)

Microscopy nuclei (BBBC039, CC0), not brain MRI. From a clean clone on one CPU core:

```sh
uv sync --locked --extra dev --extra ml
uv run --no-sync python scripts/fetch_real_data.py --output ./real-source
uv run --no-sync python scripts/quality_cpu.py --data ./real-source --output ./quality-out
```

Result of the verification run: development Dice 0.8928, test Dice 0.8964 on 16 official test fields (about 25 s of training, output `best.keras` about 6 MB, `report.json`, `test-pred.png`). Training is not bit-deterministic, so expect small differences. The test fields have been observed in earlier runs, so this is a reproducible demonstration, not an untouched benchmark. The 2-epoch `real_data.py` run below is only a wiring check (Dice about 0.07). The original brain MRI weights were never recovered.

## Real-data CPU example

See [actual results, source terms and explicit download/preparation commands](docs/real-data.md). These are small development runs, not accuracy or production guarantees. No model/data is committed.

## Measured CPU quality follow-up

See [quality experiments, results and limits](docs/quality-cpu.md). These go beyond wiring smokes; some improve measured results, while others expose overfit or a stronger simple baseline. No quality guarantee.


## Reliability follow-up

See [new ordinary-use safeguards and limits](docs/fix-pass.md). Model-quality scores are unchanged.
