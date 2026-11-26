import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from medical_segmentation.cli import app

runner = CliRunner()


def test_help_without_tensorflow() -> None:
    code = (
        "import sys;import medical_segmentation.cli;"
        "assert 'tensorflow' not in sys.modules"
    )
    assert (
        subprocess.run([sys.executable, "-c", code], capture_output=True).returncode
        == 0
    )
    assert runner.invoke(app, ["--help"]).exit_code == 0
    assert runner.invoke(app, ["train", "--help"]).exit_code == 0


def test_data_config(tmp_path: Path) -> None:
    for name in ["a", "b"]:
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(
            tmp_path / f"{name}.png"
        )
        Image.fromarray(np.ones((16, 16), dtype=np.uint8)).save(
            tmp_path / f"{name}_mask.png"
        )
    data = tmp_path / "data.csv"
    data.write_text("image,mask,patient\na.png,a_mask.png,p1\nb.png,b_mask.png,p2\n")
    config = tmp_path / ".env"
    config.write_text(f"MANIFEST_CSV={data}\n")
    result = runner.invoke(app, ["check-data", "--config", str(config), "--size", "16"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "pairs": 2,
        "train_images": 1,
        "validation_images": 1,
        "patients": 2,
    }
    config.write_text("OTHER=none\n")
    assert (
        runner.invoke(
            app, ["check-data", "--data", str(data), "--config", str(config)]
        ).exit_code
        == 0
    )
    for args in [
        [],
        ["--config", str(config)],
        ["--config", str(tmp_path / "missing")],
        ["--data", str(tmp_path / "missing")],
    ]:
        result = runner.invoke(app, ["check-data", *args])
        assert result.exit_code == 2
        assert "Traceback" not in result.output
    if importlib.util.find_spec("tensorflow") is None:
        result = runner.invoke(
            app,
            [
                "train",
                str(data),
                str(tmp_path / "out"),
                "--size",
                "16",
                "--filters",
                "1",
            ],
        )
        assert result.exit_code == 2
