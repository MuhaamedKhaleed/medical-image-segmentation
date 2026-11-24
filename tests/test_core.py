from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from medical_segmentation.core import Pair, dice, load, manifest, split


def pair(tmp_path: Path, name: str = "a", patient: str = "p1") -> Pair:
    image = tmp_path / f"{name}.png"
    mask = tmp_path / f"{name}_mask.png"
    Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16)).save(image)
    values = np.zeros((16, 16), dtype=np.uint8)
    values[4:12, 4:12] = 255
    Image.fromarray(values).save(mask)
    return Pair(image, mask, patient)


def test_manifest_load_and_patient_split(tmp_path: Path) -> None:
    a = pair(tmp_path)
    b = pair(tmp_path, "b")
    c = pair(tmp_path, "c", "p2")
    csv = tmp_path / "data.csv"
    csv.write_text(
        "image,mask,patient\na.png,a_mask.png,p1\nb.png,b_mask.png,p1\nc.png,c_mask.png,p2\n"
    )
    pairs = manifest(csv)
    assert pairs == [a, b, c]
    training, validation = split(pairs)
    assert {p.patient for p in training}.isdisjoint(p.patient for p in validation)
    assert (training, validation) == split(pairs)
    x, y = load(pairs, 32)
    assert x.shape == y.shape == (3, 32, 32, 1)
    assert x.dtype == y.dtype == np.float32
    assert 0 <= x.min() <= x.max() <= 1
    assert set(np.unique(y)) == {0, 1}
    assert int(y.sum()) == 3 * 16 * 16


@pytest.mark.parametrize(
    "text",
    [
        "",
        "image,mask\na,b\n",
        "image,mask,patient,image\na,b,c,d\n",
        "image,mask,patient\na,b,\n",
        "image,mask,patient\na,b\n",
        "image,mask,patient\na,b,c,d\n",
        "image,mask,patient\nmissing,missing_mask,p\n",
    ],
)
def test_bad_manifest(tmp_path: Path, text: str) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text(text)
    with pytest.raises(ValueError):
        manifest(csv)


def test_no_fake_or_duplicate_masks(tmp_path: Path) -> None:
    a = pair(tmp_path)
    csv = tmp_path / "data.csv"
    for text in [
        "image,mask,patient\na.png,a.png,p1\n",
        "image,mask,patient\na.png,a_mask.png,p1\na.png,a_mask.png,p2\n",
        "image,mask,patient\n",
    ]:
        csv.write_text(text)
        with pytest.raises(ValueError):
            manifest(csv)
    a.mask.unlink()
    with pytest.raises(FileNotFoundError):
        load([a])
    assert not a.mask.exists()


@pytest.mark.parametrize("fraction", [0, 1, -1, float("nan"), float("inf")])
def test_bad_split(tmp_path: Path, fraction: float) -> None:
    a = pair(tmp_path)
    with pytest.raises(ValueError):
        split([a], fraction)


def test_mask_guards(tmp_path: Path) -> None:
    a = pair(tmp_path)
    for values in [
        np.zeros((8, 8), dtype=np.uint8),
        np.full((16, 16), 128, dtype=np.uint8),
        np.zeros((16, 16, 3), dtype=np.uint8),
    ]:
        Image.fromarray(values).save(a.mask)
        with pytest.raises(ValueError):
            load([a])
    for size in [0, 15, 17, 1024]:
        with pytest.raises(ValueError):
            load([a], size)
    with pytest.raises(ValueError):
        load([])
    a.mask.write_text("not an image")
    with pytest.raises(OSError):
        load([a])


def test_real_empty_mask_and_zero_one_encoding(tmp_path: Path) -> None:
    a = pair(tmp_path)
    for value in [0, 1]:
        Image.fromarray(np.full((16, 16), value, dtype=np.uint8)).save(a.mask)
        _, y = load([a], 16)
        assert set(np.unique(y)) == {value}


def test_dice_guards() -> None:
    y = np.array([0, 1, 1, 0], dtype=np.float32)
    assert dice(y, y) == 1
    assert dice(y, 1 - y) == 0
    assert dice(np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)) == 1
    assert dice(y, np.array([0, 1, 0, 0], dtype=np.float32)) == pytest.approx(2 / 3)
    for a, b in [(y, y[:2]), (y, y * np.nan), (y, y + 1), (y + 1, y), (y[:0], y[:0])]:
        with pytest.raises(ValueError):
            dice(a, b)
