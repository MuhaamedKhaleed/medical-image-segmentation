"""Strict image/mask manifests, patient-group split and binary-mask loading."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image


@dataclass(frozen=True)
class Pair:
    """One local image, genuine mask, and nonempty patient-group identifier."""

    image: Path
    mask: Path
    patient: str


def manifest(path: Path) -> list[Pair]:
    """Read image,mask,patient CSV relative to CSV folder, reject repeats."""
    result = []
    images = set()
    masks = set()
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        headers = rows.fieldnames or []
        if len(headers) != len(set(headers)) or not {"image", "mask", "patient"} <= set(
            headers
        ):
            raise ValueError("CSV requires unique image,mask,patient headers.")
        for i, row in enumerate(rows, 2):
            if None in row or any(v is None or not v.strip() for v in row.values()):
                raise ValueError(f"Missing or malformed fields at row{i}.")
            image = (path.parent / row["image"].strip()).resolve()
            mask = (path.parent / row["mask"].strip()).resolve()
            if image == mask:
                raise ValueError("Image cannot be its own segmentation mask.")
            if not image.is_file() or not mask.is_file():
                raise ValueError(f"Missing image or genuine mask at row{i}.")
            if image in images or mask in masks:
                raise ValueError(f"Repeated image or mask at row{i}.")
            images.add(image)
            masks.add(mask)
            result.append(Pair(image, mask, row["patient"].strip()))
    if not result:
        raise ValueError("Manifest contains no pairs.")
    return result


def split(
    pairs: list[Pair], fraction: float = 0.2, seed: int = 42
) -> tuple[list[Pair], list[Pair]]:
    """Split by explicit patient ID, not image slices. At least two patients."""
    groups = sorted({p.patient for p in pairs})
    if len(groups) < 2 or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Need two patient groups and validation fraction in (0,1).")
    random.Random(seed).shuffle(groups)
    n = max(1, min(len(groups) - 1, math.ceil(len(groups) * fraction)))
    val = set(groups[:n])
    return [p for p in pairs if p.patient not in val], [
        p for p in pairs if p.patient in val
    ]


def load(
    pairs: list[Pair], size: int = 128
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Read genuine masks, require native matching dimensions and 0/1 or0/255.

    Images resize bilinearly; categorical masks use nearest-neighbor. Black
    masks are valid only when supplied explicitly, never manufactured.
    """
    if not pairs or size < 16 or size > 512 or size % 16:
        raise ValueError("Nonempty pairs and size16..512 divisible by16 required.")
    images = []
    masks = []
    for pair in pairs:
        with Image.open(pair.image) as image, Image.open(pair.mask) as mask:
            if image.size != mask.size:
                raise ValueError("Image/mask dimensions differ before resize.")
            raw = np.asarray(mask)
            if raw.ndim != 2 or not (
                set(np.unique(raw)) <= {0, 1} or set(np.unique(raw)) <= {0, 255}
            ):
                raise ValueError(
                    "Mask must be single-channel binary0/1 or0/255, "
                    "not an image or soft labels."
                )
            images.append(
                np.asarray(
                    image.convert("L").resize((size, size), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
                / 255
            )
            masks.append(
                (
                    np.asarray(mask.resize((size, size), Image.Resampling.NEAREST)) > 0
                ).astype(np.float32)
            )
    return np.stack(images)[..., None], np.stack(masks)[..., None]


def dice(target: NDArray[np.float32], prediction: NDArray[np.float32]) -> float:
    """Global thresholded Dice with empty/empty=1, explicit shape/range guards."""
    if (
        target.shape != prediction.shape
        or not target.size
        or not np.isfinite(prediction).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("Dice requires matching finite nonempty arrays.")
    if (
        not np.isin(target, [0, 1]).all()
        or np.any(prediction < 0)
        or np.any(prediction > 1)
    ):
        raise ValueError("Dice requires binary targets and probabilities in [0,1].")
    a = target > 0
    b = prediction >= 0.5
    den = int(a.sum() + b.sum())
    return 1.0 if den == 0 else float(2 * np.logical_and(a, b).sum() / den)
