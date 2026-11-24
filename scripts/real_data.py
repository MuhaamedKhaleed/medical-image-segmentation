"""Explicit real-data CPU development recipe. Never runs on import."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared local source directory")
    parser.add_argument(
        "--base-model", help="Trusted compatible local DistilBERT directory"
    )
    parser.add_argument(
        "--manifest", help="Reviewed selected Open Images source manifest"
    )
    args = parser.parse_args()
    import io
    import json
    import os
    import pathlib
    import time
    import zipfile

    for k, v in {
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "-1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
    }.items():
        os.environ[k] = v
    import numpy as np
    from PIL import Image

    from medical_segmentation.core import Pair
    from medical_segmentation.model import train

    root = pathlib.Path(args.data)
    out = root / "prepared"
    out.mkdir()
    training = []
    validation = []
    with (
        zipfile.ZipFile(root / "metadata.zip") as meta,
        zipfile.ZipFile(root / "images.zip") as images,
        zipfile.ZipFile(root / "masks.zip") as masks,
    ):
        for group, count, target in [
            ("training", 16, training),
            ("validation", 8, validation),
        ]:
            for name in (
                meta.read("metadata/" + group + ".txt").decode().splitlines()[:count]
            ):
                name = name.strip()
                stem = pathlib.Path(name).stem
                a = np.asarray(
                    Image.open(io.BytesIO(images.read("images/" + stem + ".tif")))
                ).astype(np.float32)
                # Percentile scaling for 16-bit microscopy, documented conversion not
                # fabricated labels.
                lo, hi = np.percentile(a, [1, 99])
                a = np.clip((a - lo) / max(1, hi - lo) * 255, 0, 255).astype("uint8")
                m = np.asarray(Image.open(io.BytesIO(masks.read("masks/" + name))))
                # Source instance masks use black background; merge all annotated
                # instances to binary foreground.
                fg = np.any(m[:, :, :3] != 0, axis=-1) if m.ndim == 3 else m != 0
                Image.fromarray(a).save(out / (stem + ".png"))
                Image.fromarray((fg * 255).astype("uint8")).save(
                    out / (stem + "-mask.png")
                )
                target.append(
                    Pair(
                        out / (stem + ".png"),
                        out / (stem + "-mask.png"),
                        group + "-field-" + stem,
                    )
                )
    start = time.time()
    scores = train(
        training, validation, root / "model-real", epochs=2, size=64, filters=2
    )
    report = {
        "real_data": True,
        "dataset": "BBBC039v1 U2OS fluorescence nuclei",
        "source": "https://bbbc.broadinstitute.org/BBBC039",
        "license": "CC0",
        "train_fields": 16,
        "validation_fields": 8,
        "split": (
            "first official training16 and validation8; official test "
            "untouched; fields not patients"
        ),
        "transforms": (
            "16-bit image1/99percentile scaling; real RGB instance "
            "masks merged to binary foreground, nearest-neighbor "
            "resizing64"
        ),
        "training": scores,
        "elapsed_seconds": round(time.time() - start, 2),
        "limits": (
            "microscopy nuclei, not brain MRI or patient clinical "
            "validation; small2epoch CPU development run"
        ),
    }
    (root / "real-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    canvas = Image.new("RGB", (384, 160), "white")
    for i, p in enumerate(validation[:3]):
        canvas.paste(Image.open(p.image).resize((128, 128)), (i * 128, 0))
        canvas.paste(
            Image.open(p.mask).resize((128, 128)), (i * 128, 0)
        ) if False else None
    canvas.save(str(root / "real-microscopy-inputs.png"))


if __name__ == "__main__":
    main()
