import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from medical_segmentation.cli import app
from medical_segmentation.core import load, manifest, split
from medical_segmentation.model import build, train

tf = pytest.importorskip("tensorflow")


def test_actual_unet_training_save_reload(tmp_path: Path) -> None:
    for i in range(3):
        values = np.zeros((16, 16), dtype=np.uint8)
        values[4:12, 4:12] = 255
        Image.fromarray(values).save(tmp_path / f"x{i}.png")
        Image.fromarray(values).save(tmp_path / f"m{i}.png")
    csv = tmp_path / "data.csv"
    csv.write_text(
        "image,mask,patient\nx0.png,m0.png,p0\nx1.png,m1.png,p1\nx2.png,m2.png,p2\n"
    )
    training, validation = split(manifest(csv))
    x, _ = load(validation, 16)
    model = build(16, 1)
    assert model(x, training=False).shape == (1, 16, 16, 1)
    assert len(model.trainable_variables) > 0
    scores = train(training, validation, tmp_path / "artifact", 1, 16, 1)
    assert scores["validation_images"] == 1 and 0 <= scores["validation_dice"] <= 1
    assert np.isfinite(scores["validation_loss"])
    restored = tf.keras.models.load_model(
        tmp_path / "artifact" / "model.keras", safe_mode=True
    )
    assert restored(x, training=False).shape == (1, 16, 16, 1)
    assert json.loads((tmp_path / "artifact" / "validation.json").read_text()) == scores
    result = CliRunner().invoke(
        app,
        ["train", str(csv), str(tmp_path / "cli"), "--size", "16", "--filters", "1"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["validation_patients"] == 1
    with pytest.raises(ValueError):
        train(training, validation, tmp_path / "artifact", 1, 16, 1)
    with pytest.raises(ValueError):
        train(training, training, tmp_path / "new", 1, 16, 1)
    with pytest.raises(ValueError):
        train(training, validation, tmp_path / "new", 0, 16, 1)


@pytest.mark.parametrize("size,filters", [(15, 1), (16, 0), (16, 65), (528, 1)])
def test_bad_model(size: int, filters: int) -> None:
    with pytest.raises(ValueError):
        build(size, filters)
