"Explicit measured CPU quality experiment. No import-time work or downloads."


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model", type=Path, help="Translation only: trusted local Marian"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new directory.")
    ("Measured CPU development experiment, preserves official test for one final read.")
    import os

    for k in ["TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS"]:
        os.environ[k] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import io
    import json
    import resource
    import time
    import zipfile
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from medical_segmentation.core import Pair, dice, load
    from medical_segmentation.model import build

    root = Path(str(args.output))
    root.mkdir(exist_ok=False)
    source = args.data
    splits = {}
    with (
        zipfile.ZipFile(source / "metadata.zip") as meta,
        zipfile.ZipFile(source / "images.zip") as images,
        zipfile.ZipFile(source / "masks.zip") as masks,
    ):
        for group, count in [("training", 64), ("validation", 16), ("test", 16)]:
            pairs = []
            names = (
                meta.read("metadata/" + group + ".txt").decode().splitlines()[:count]
            )
            for name in names:
                name = name.strip()
                stem = Path(name).stem
                a = np.asarray(
                    Image.open(io.BytesIO(images.read("images/" + stem + ".tif")))
                ).astype(np.float32)
                lo, hi = np.percentile(a, [1, 99])
                a = np.clip((a - lo) / max(1, hi - lo) * 255, 0, 255).astype("uint8")
                m = np.asarray(Image.open(io.BytesIO(masks.read("masks/" + name))))
                fg = np.any(m[:, :, :3] != 0, axis=-1) if m.ndim == 3 else m != 0
                Image.fromarray(a).save(root / (stem + ".png"))
                Image.fromarray((fg * 255).astype("uint8")).save(
                    root / (stem + "-mask.png")
                )
                pairs.append(
                    Pair(
                        root / (stem + ".png"),
                        root / (stem + "-mask.png"),
                        group + "-" + stem,
                    )
                )
            splits[group] = pairs
    x, y = load(splits["training"], 64)
    vx, vy = load(splits["validation"], 64)
    model = build(64, 8, 42)
    model.optimizer.learning_rate.assign(0.001)
    start = time.time()
    history = []
    best = -1
    for epoch in range(40):
        order = np.random.default_rng(42 + epoch).permutation(len(x))
        for i in range(0, len(x), 8):
            idx = order[i : i + 8]
            model.train_on_batch(x[idx], y[idx])
        pred = model(vx, training=False).numpy()
        score = dice(vy, pred)
        row = {"epoch": epoch + 1, "development_dice": score}
        history.append(row)
        if score > best:
            best = score
            model.save(root / "best.keras")
        if epoch in [0, 4, 9, 19, 39]:
            print(row, flush=True)
    import tensorflow as tf

    model = tf.keras.models.load_model(root / "best.keras")
    tx, ty = load(splits["test"], 64)
    pred = model(tx, training=False).numpy()
    test = dice(ty, pred)
    report = {
        "dataset": "BBBC039v1",
        "source": "https://bbbc.broadinstitute.org/BBBC039",
        "split": (
            "64official train/16official development/16first official "
            "test fields; not patients"
        ),
        "size": 64,
        "filters": 8,
        "epochs": 40,
        "learning_rate": 0.001,
        "best_development_dice": best,
        "test_dice": test,
        "elapsed_seconds": time.time() - start,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "history": history,
        "limits": (
            "small field subsets, test read once after dev selection; not"
            " clinical MRI, not independent-site validation"
        ),
    }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}))
    Image.fromarray((pred[0, :, :, 0] * 255).astype("uint8")).save(
        root / "test-pred.png"
    )


if __name__ == "__main__":
    main()
