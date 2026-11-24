"""Explicit source download. No import-time network or data redistribution."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="New source directory"
    )
    parser.add_argument(
        "--cached", type=Path, help="Use already downloaded files, without network"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for filename, url, sha in SOURCES:
        target = args.output / filename
        if args.cached:
            shutil.copyfile(args.cached / filename, target)
        else:
            with (
                urllib.request.urlopen(url, timeout=120) as source,
                target.open("wb") as out,
            ):
                shutil.copyfileobj(source, out)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != sha:
            target.unlink()
            raise ValueError("Source checksum mismatch: " + filename)
        manifest.append({"file": filename, "url": url, "sha256": actual})
    (args.output / "source-manifest.json").write_text(json.dumps(manifest, indent=2))


SOURCES = [
    (
        "images.zip",
        "https://data.broadinstitute.org/bbbc/BBBC039/images.zip",
        "6f30a5d4fe38c928ded972704f085975f8dc0d65d9aa366df00e5a9d449fddd7",
    ),
    (
        "masks.zip",
        "https://data.broadinstitute.org/bbbc/BBBC039/masks.zip",
        "f9e6043d8ca56344a4886f96a700d804d6ee982f31e2b2cd3194af2a053c2710",
    ),
    (
        "metadata.zip",
        "https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip",
        "a2c1f900bed9ba92a99553efd4c2ae98598433691c7401d818653ab61110deb2",
    ),
]

if __name__ == "__main__":
    main()
