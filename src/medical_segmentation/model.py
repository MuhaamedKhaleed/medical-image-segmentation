"""Explicit CPU U-Net training. No downloads or automatic fitting."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .core import Pair, dice, load


def backend() -> Any:
    """Lazy optional TensorFlow import, use CPU only."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ValueError("Install ML extra with uv sync --extra ml.") from exc
    tf.config.set_visible_devices([], "GPU")
    return tf


def build(size: int = 128, filters: int = 64, seed: int = 42) -> Any:
    """Original four-level U-Net widths, optional smaller width for smoke tests."""
    if size < 16 or size > 512 or size % 16 or not 1 <= filters <= 64:
        raise ValueError("Size16..512 divisible by16 and filters1..64 required.")
    tf = backend()
    tf.keras.utils.set_random_seed(seed)
    layers = tf.keras.layers
    inputs = layers.Input((size, size, 1))
    x = inputs
    skips = []
    for level in range(4):
        width = filters * 2**level
        x = layers.Conv2D(width, 3, activation="relu", padding="same")(x)
        x = layers.Conv2D(width, 3, activation="relu", padding="same")(x)
        if level == 3:
            x = layers.Dropout(0.5)(x)
        skips.append(x)
        x = layers.MaxPooling2D()(x)
    for _ in range(2):
        x = layers.Conv2D(filters * 16, 3, activation="relu", padding="same")(x)
    x = layers.Dropout(0.5)(x)
    for level in reversed(range(4)):
        width = filters * 2**level
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(width, 2, activation="relu", padding="same")(x)
        x = layers.Concatenate()([skips[level], x])
        for _ in range(2):
            x = layers.Conv2D(width, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(2, 3, activation="relu", padding="same")(x)
    output = layers.Conv2D(1, 1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy")
    return model


def train(
    training: list[Pair],
    validation: list[Pair],
    output: Path,
    epochs: int = 1,
    size: int = 128,
    filters: int = 64,
    seed: int = 42,
) -> dict[str, float | int]:
    """Train explicitly, validate genuine masks, save only finite final metrics."""
    if output.exists():
        raise ValueError("Output exists; choose a new directory.")
    if epochs < 1:
        raise ValueError("Epochs must be positive.")
    if {p.patient for p in training} & {p.patient for p in validation}:
        raise ValueError("Patient overlap across splits.")
    if {p.image for p in training} & {p.image for p in validation}:
        raise ValueError("Image overlap across splits.")
    x, y = load(training, size)
    vx, vy = load(validation, size)
    model = build(size, filters, seed)
    for _ in range(epochs):
        for start in range(0, len(x), 4):
            loss = float(
                model.train_on_batch(x[start : start + 4], y[start : start + 4])
            )
            if not math.isfinite(loss):
                raise ValueError("Nonfinite training loss, no artifact saved.")
    prediction = model(vx, training=False).numpy()
    loss = float(model.evaluate(vx, vy, verbose=0))
    if not math.isfinite(loss):
        raise ValueError("Nonfinite validation loss, no artifact saved.")
    scores = {
        "validation_loss": loss,
        "validation_dice": dice(vy, prediction),
        "validation_images": len(vx),
        "validation_patients": len({p.patient for p in validation}),
    }
    output.mkdir(parents=True)
    model.save(output / "model.keras")
    (output / "validation.json").write_text(json.dumps(scores))
    return scores
