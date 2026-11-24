"""Explicit synthetic wiring demo. No real-data accuracy or quality claim."""

import json
import os
import sys
from pathlib import Path


def main() -> None:
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "-1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }.items():
        os.environ.setdefault(key, value)

    out = Path(sys.argv[1] if len(sys.argv) == 2 else "synthetic-run")
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python scripts/demo.py [NEW_OUTPUT_DIRECTORY]")
    out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import tensorflow as tf
    from PIL import Image

    from medical_segmentation.core import load, manifest, split
    from medical_segmentation.model import train

    tf.keras.utils.set_random_seed(42)
    for i in range(3):
        a = np.zeros((16, 16), dtype=np.uint8)
        a[4:12, 4:12] = 255
        Image.fromarray(a).save(out / f"x{i}.png")
        Image.fromarray(a).save(out / f"m{i}.png")
    path = out / "manifest.csv"
    path.write_text(
        "image,mask,patient\n" + "".join(f"x{i}.png,m{i}.png,p{i}\n" for i in range(3))
    )
    training, validation = split(manifest(path))
    scores = train(training, validation, out / "model", 1, 16, 1)
    x, _ = load(validation, 16)
    model = tf.keras.models.load_model(out / "model/model.keras", safe_mode=True)
    p = model(x, training=False).numpy()[0, :, :, 0]
    Image.fromarray((p * 255).astype("uint8")).save(out / "prediction.png")
    assert p.shape == (16, 16) and np.isfinite(p).all()
    result = {"training": scores, "prediction_shape": list(p.shape)}
    scope = "invented_squares_not_MRI_not_clinical_or_diagnostic"
    report = {
        "synthetic_only": True,
        "scope": scope,
        "quality_claim": False,
        "result": result,
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"synthetic_only": True, "report": str(out / "report.json")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
