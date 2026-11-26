# Real-data CPU development run

Source: https://bbbc.broadinstitute.org/BBBC039 . Actual recorded result: `real-data-output.json`. This is a small development run, not full training, production readiness or an untouched benchmark. No data/model artifacts are committed. Run from the repository root after installing the locked ML/dev extras. This script starts real CPU training only when explicitly invoked. Outputs use fixed new names under the supplied data folder and refuse rerunning into an existing artifact. Use a new folder for another run.

```sh
uv sync --locked --extra dev --extra ml
uv run --no-sync python scripts/real_data.py --data /path/to/prepared-source
```

Download `images.zip`, `masks.zip` and `metadata.zip` from links on the source BBBC039 page. It explicitly labels BBBC039v1 CC0. Actual URLs: https://data.broadinstitute.org/bbbc/BBBC039/images.zip , https://data.broadinstitute.org/bbbc/BBBC039/masks.zip , https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip . About78MB image archive plus masks. Script reads exact members without extracting paths, first16 official train/8 dev fields, preserves real annotated foreground, merges instance colors to binary and scales16bit images at percentiles1/99. This is nuclei microscopy, not MRI or patients. `validation_patients` in the current generic API is field groups for this dataset. Official test untouched. Low Dice is reported, not hidden.

## Actual limits

microscopy nuclei, not brain MRI or patient clinical validation; small2epoch CPU development run; API key validation_patients means eight field groups here, not real patients

## Explicit download and preparation

The optional helper downloads only the documented primary-source files, checks SHA-256 and refuses an existing destination. It never runs during installation, import or tests. Source attribution/license cautions above still apply. No model/data is bundled. For an offline copy, add `--cached /path/to/source-files` with the same filenames; checksums are still required.

```sh
uv run --no-sync python scripts/fetch_real_data.py --output ./real-source
uv run --no-sync python scripts/real_data.py --data ./real-source
```

The hashes pin the exact small development sources used here, not a permission warranty or a quality benchmark. If a source changes, the helper fails rather than silently training on different data.

## Docker without a registry account

This repository supplies a Dockerfile, not a published registry image. Build locally once. The default remains the small invented-input wiring demo. No account, paid service or automatic source download is required by the image.

```sh
docker build -f Dockerfile.ml -t project-ml .
docker run --rm --network none project-ml
```

The image also includes `/app/scripts`. After explicit source preparation on the host, mount a new writable working directory and invoke the real script with `--entrypoint python`. Keep model/source mounts read-only when the script does not write beside them. The image uses uid10001, so grant that uid write access to the output folder, do not run the container as root to bypass this. Data/preprocessing scripts currently place artifacts beside source files; mount the working copy read-write and keep your original untouched. The environment disables Hugging Face downloads. Network-none is suitable only after sources/models are local; an explicit fetch helper needs network by design. CPU/dependency builds were checked on GitHub's Linux runner, not every laptop OS.
