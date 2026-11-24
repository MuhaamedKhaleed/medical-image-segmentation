"""Thin research CLI. No clinical inference or automatic training."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer
from dotenv import dotenv_values

from .core import load, manifest, split

app = typer.Typer(
    help="Research-only genuine image/mask checks and explicit CPU U-Net training."
)
logger = logging.getLogger(__name__)


def run(action: Callable[[], Any]) -> None:
    """Keep machine-readable output separate from actionable input errors."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        typer.echo(json.dumps(action()))
    except (ValueError, OSError, TypeError) as exc:
        logger.error("%s", exc)
        raise typer.Exit(2) from exc


def resolve(data: Path | None, config: Path | None) -> Path:
    """Resolve explicit data or opt-in MANIFEST_CSV dotenv setting."""
    if data:
        return data
    if config:
        if not config.is_file():
            raise ValueError("Config missing. Supply existing dotenv.")
        value = dotenv_values(config).get("MANIFEST_CSV")
        if value:
            return Path(value)
    raise ValueError("Supply --data manifest.csv or --config containing MANIFEST_CSV.")


@app.command()
def check_data(
    data: Path | None = None,
    config: Path | None = None,
    size: int = 128,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> None:
    """Verify all pairs and patient-disjoint split, never fabricate masks."""

    def action() -> dict[str, int]:
        pairs = manifest(resolve(data, config))
        training, validation = split(pairs, validation_fraction, seed)
        load(pairs, size)
        return {
            "pairs": len(pairs),
            "train_images": len(training),
            "validation_images": len(validation),
            "patients": len({p.patient for p in pairs}),
        }

    run(action)


@app.command()
def train(
    data: Path,
    output: Path,
    epochs: int = 1,
    size: int = 128,
    filters: int = 64,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> None:
    """Explicit CPU training on genuine masks. Not a diagnostic device."""

    def action() -> dict[str, float | int]:
        from .model import train as fit

        training, validation = split(manifest(data), validation_fraction, seed)
        return fit(training, validation, output, epochs, size, filters, seed)

    run(action)
