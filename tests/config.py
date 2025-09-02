from pathlib import Path
import pytest
from planeclf.config import Paths, Config

def test_paths_singleton_basic(tmp_path):
    p = Paths.get()
    assert p.root == tmp_path.resolve()
    assert p.data_dir.name == "data"          # .../data_raw/fgvc-aircraft-2013b/**data**
    assert p.models_dir.name == "models"
    assert p.wording_dir.as_posix().endswith("src/planeclf")
    assert p.assets_dir.name == "assets"

def test_paths_reconfigure_same_root_ok(tmp_path):
    p1 = Paths.get()
    p2 = Paths.set(tmp_path)  # même racine -> ok
    assert p1 is p2

def test_config_loaded_values():
    c = Config.get()
    assert c.img_width == 64 and c.img_height == 64
    assert isinstance(c.seed, int)
    d = c.to_dict()
    assert d["img_width"] == 64 and "models_dir" in d

def test_config_missing_yaml_raises(tmp_path, monkeypatch):
    # changer de répertoire vers un workspace sans app.yaml
    other = tmp_path / "other"
    other.mkdir(exist_ok=True)
    monkeypatch.chdir(other)
    Config.reset()
    with pytest.raises(FileNotFoundError):
        Config.set()
