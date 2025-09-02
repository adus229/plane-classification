from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from planeclf.config import Paths, Config

@pytest.fixture(autouse=True)
def _workspace(tmp_path: Path, monkeypatch):
    # Travailler dans un dossier éphémère
    monkeypatch.chdir(tmp_path)
    # Arbo minimale
    (tmp_path / "src" / "planeclf").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    # app.yaml minimal pour ton Config actuel
    (tmp_path / "src" / "planeclf" / "app.yaml").write_text(
        "image_width: 64\nimage_height: 64\nseed: 42\n",
        encoding="utf-8",
    )
    # Reset & init des singletons
    Paths.reset(); Config.reset()
    Paths.set(tmp_path)
    Config.set()
    yield
    Paths.reset(); Config.reset()

@pytest.fixture
def png_file(tmp_path: Path) -> Path:
    p = tmp_path / "tmp.png"
    arr = np.full((8, 8, 3), 128, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(p)  # vrai PNG sur disque
    return p
