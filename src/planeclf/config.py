from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional, Tuple, Dict, Any
import yaml


@dataclass(frozen=True, init=False)
class Paths:
    """Singleton holding project paths (simple, non-thread-safe).

    Usage:
        Paths.set(root=".")      # do once (optional)
        p = Paths.get()          # anywhere later
    """
    root: Path = Path
    data_dir: Path
    models_dir: Path
    wording_dir: Path = field(init=False)
    assets_dir: Path = field(init=False)

    _instance: ClassVar[Optional["Paths"]] = None

    def __init__(self, root: Path | str = "."):
        # Initialize only if not already set
        if type(self)._instance is not None:
            return
        root = Path(root).resolve()

        object.__setattr__(self, "root", root)
        object.__setattr__(self, "data_dir", root / "data_raw/fgvc-aircraft-2013b/data")
        object.__setattr__(self, "models_dir", root / "models")
        object.__setattr__(self, "wording_dir", root / "src" / "planeclf")
        object.__setattr__(self, "assets_dir", root / "assets")
        type(self)._instance = self

    # ---- Public API ----
    @classmethod
    def set(cls, root: Path | str = ".") -> "Paths":
        """Initialize the singleton if needed, or validate same root."""
        if cls._instance is None:
            return cls(root)
        # If already initialized, refuse changing the root to prevent ambiguity.
        new_root = Path(root).resolve()
        if cls._instance.root != new_root:
            raise RuntimeError(f"Paths already initialized with root={cls._instance.root}, "
                               f"cannot change to {new_root}")
        return cls._instance

    @classmethod
    def get(cls) -> "Paths":
        """Return the instance (initializes with '.' if first use)."""
        return cls._instance or cls(".")

    @classmethod
    def get_train_paths(cls, target="manufacturer") -> Tuple[Path, Path]:
        """Return the train and validation CSV paths."""
        paths = cls.get()

        return {"train":  paths.data_dir/f"images_{target}_train.txt",
                "val": paths.data_dir/f"images_{target}_val.txt"}

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (handy for tests)."""
        cls._instance = None


# ---------- Minimal singleton: Config ----------

@dataclass(frozen=True, init=False)
class Config:
    """Singleton holding app configuration loaded from YAML (simple, non-thread-safe).

    Usage:
        Paths.set(root=".")          # optional
        Config.set()                 # reads <root>/src/planeclf/app.yaml by default
        cfg = Config.get()

    Attributes
    ----------
    paths : Paths
    img_size : (int, int) = (width, height)
    batch_size : int
    epochs : int
    target : str
    val_ratio : float
    seed : int
    img_width : int
    img_height : int
    """
    paths: Paths
    img_size: Tuple[int, int]
    target: str
    val_ratio: float
    seed: int = field(default=42)
    img_width: int = field(init=False)
    img_height: int = field(init=False)

    _instance: ClassVar[Optional["Config"]] = None

    def __init__(self,):
        if type(self)._instance is not None:
            return
        config_path = Path("src/planeclf/app.yaml")

        paths = Paths.get()
        if not config_path.exists():
            raise FileNotFoundError(f"Config YAML not found: {config_path}")

        cfg: Dict[str, Any] = yaml.safe_load(
            config_path.read_text(encoding="utf-8")) or {}

        img_w = int(cfg["image_width"])
        img_h = int(cfg["image_height"])
        seed = int(cfg.get("seed", 42))

        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "img_size", (img_w, img_h))
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "img_width", img_w)
        object.__setattr__(self, "img_height", img_h)

        type(self)._instance = self

    # ---- Public API ----
    @classmethod
    def set(cls) -> "Config":
        """Initialize from YAML if needed; refuses reconfig with different file."""
        if cls._instance is None:
            return cls()
        # If called again, we just return the existing instance.
        return cls._instance

    @classmethod
    def get(cls) -> "Config":
        """Return the instance (initializes with default YAML if first use)."""
        return cls._instance or cls()

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (handy for tests)."""
        cls._instance = None

    # Convenience
    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": str(self.paths.root),
            "data_dir": str(self.paths.data_dir),
            "models_dir": str(self.paths.models_dir),
            "img_size": self.img_size,
            "img_width": self.img_width,
            "img_height": self.img_height,
            "seed": self.seed,
        }
