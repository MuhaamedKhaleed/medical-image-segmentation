# Verification and limits

## Original defects and deliberate changes

The notebook's loader manufactured zero masks for missing annotations. It used bilinear mask resize, mixed unverified mask augmentation, and assumed a mask directory after a classification-data download. These are not valid segmentation labels. The repaired historical notebook refuses missing/unreadable masks, uses nearest-neighbor resize and removes unverified augmentation/download/credential cells. Outputs were cleared. Full notebook execution remains unverified.

The package requires local manifest image,mask,patient rows, genuine binary masks and patient-disjoint validation. It does not infer annotation provenance or patient identity. Users must verify IDs and annotations. Images and masks must have equal native dimensions. Explicit empty masks are valid; fabricated ones are not. Paths differ, but identical content under different paths is not detected. It is a research pipeline, not clinical software.

The U-Net retains the original four encoder/decoder levels, default base64/bottleneck1024 widths, dropout and final2-channel convolution before sigmoid. Tests/demo reduce base width to1 and input32/16 for resource-bounded wiring verification. This does not validate default-width convergence or quality. CPU training is explicit; no GPU assumption, auto-download, secret or dataset upload.

## Exact local checks

- `uv sync --locked --extra dev --extra ml`: installed scoped locked environment.
- `TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 uv run pytest --cov=medical_segmentation --cov-report=term-missing`:24 passed,96% coverage.
- `uv run ruff check src tests`, `uv run ruff format --check src tests`, `uv run mypy src`: passed.
- `uv run pre-commit run --all-files`: passed.
- `uv run python scripts/synthetic_demo.py`: invented three square image/mask pairs.
- `uv run medical-segmentation check-data --data synthetic-demo/manifest.csv --size 32`:3pairs,2train,1validation,3patients.
- `uv run medical-segmentation check-data --config .env.example --size 32`:same result.
- `TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 uv run medical-segmentation train synthetic-demo/manifest.csv synthetic-demo/trained --epochs 1 --size 32 --filters 1`:finite synthetic loss/Dice, saved local artifact, no quality claim.
- Module and console `--help`: passed.

Tests include actual TensorFlow U-Net CPU forward, training batch, `.keras` save/load in safe mode, explicit CLI training, categorical mask resize, patient splits, malformed/missing files and metric conventions. A light locked environment excludes TensorFlow and explicitly skips the ML test file. No network or clinical dataset required.

Docker build and help/mounted-data checks are run by hosted CI, not a local daemon. Docker contains NumPy/Pillow and the data CLI, no TensorFlow/dev packages. CI uses pip/direct pins, local uv uses the scoped transitive lock. Ordinary public Linux runner only, no paid services or full real-data training.

## Phase accounting

Installable typed src package, explicit entry points, separate data/model/CLI responsibilities; useful validation/errors/logging; opt-in dotenv example; meaningful runtime/dev/ML groups and uv lock; actual offline tests; Ruff/mypy/hooks; public-only CI; non-root multi-stage data image; real usage/output and architecture; changelog and separate historical bugfix commit; copy-only presentation notes. Review branch only. No license changes, main merge, release, tags or fabricated history. Remote web-editor commits do not pretend a phase-perfect local history.

## Metrics and missing evidence

Global Dice thresholds at0.5 and defines empty/empty as1. It is not per-patient mean Dice, lesion sensitivity, clinical specificity, calibration or an external benchmark. Real MRI data, annotation verification, untouched test cohort, full default-width fitting, clinical validity and production serving are all unverified. Artifacts are not written transactionally if a save fails. Patient images/masks can contain private data, do not commit them.

Dataset-context sources checked:
- https://pub.towardsai.net/an-offbeat-approach-to-brain-tumor-classification-using-computer-vision-19c9e7b84664 describes the original source as brain-tumor classification images.
- https://github.com/susuter/MRI_data describes yes/no classification folders derived from that source. Neither establishes genuine pixel masks for this project.


## Gap-closure follow-up (24 November 2026)

The CI source now uses uv 0.13.8 and the committed transitive lock, rather than a new pip resolution of direct pins. Dependency installer caching is keyed by uv.lock. The local matching command is `uv sync --locked --extra dev --extra ml`, followed by the existing tests/quality commands. This does not cache or publish private datasets, weights or model outputs. Public-repository-only guards remain. The changed workflow's hosted outcome must be checked after publication; prior green runs do not validate a changed workflow.

A complete copy-only project-focused profile README draft is appended to presentation.md. No profile account edits, main merge, PR, release or license change is implied by that text. Historical notebook/full-data/quality limitations remain open, not replaced by a passing synthetic test.

## Standalone synthetic demo and CPU container follow-up

`python scripts/demo.py NEW_DIRECTORY` ran successfully in the locked local environment, with actual backend training/artifact/reload/inference where a model backend exists. `docs/demo-output.json` records one actual run, labeled synthetic-only with no quality claim. A subprocess regression checks the report and no-overwrite behavior. Tests, lint/format and package mypy rerun locally. OCR demo is decoder-only, monitoring uses actual SciPy. `Dockerfile.ml` is opt-in where applicable; hosted build/run evidence must be checked after publication, not inferred from local Python.
