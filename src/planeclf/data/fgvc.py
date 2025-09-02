from __future__ import annotations
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.request import urlretrieve
from src.planeclf.config import Config
from src.planeclf.data.fgvc import download_fgvc_aircraft


def download_fgvc_aircraft( out_path: Path | None = None) -> Path:
    """Download the FGVC Aircraft 2013b archive from Oxford VGG.

    Parameters
    ----------
    url : str | None
        Direct URL to the tarball. If None, uses the official URL.
    out_path : Path | None
        Destination file path. If None, defaults to ``data_raw/fgvc-aircraft-2013b.tar.gz``.

    Returns
    -------
    Path
        Path to the downloaded archive.
    """
    config = Config.get()
    if out_path is None:
        out_path = Path("data_raw/fgvc-aircraft-2013b.tar.gz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(config.data_url, out_path)
    return out_path


def extract_archive(archive_path: Path, dest_dir: Path) -> Path:
    """Extract a .tar(.gz|.bz2|.xz) or .zip into ``dest_dir``.

    Returns
    -------
    Path
        The extraction destination directory.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = archive_path.name.lower()
    if name.endswith((".tar.gz", ".tgz", ".tar", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")):
        with tarfile.open(archive_path, "r:*") as t:
            t.extractall(dest_dir)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    return dest_dir


def _read_mapping_file(txt: Path) -> List[Tuple[str, str]]:
    """Read FGVC mapping file (e.g., images_variant_trainval.txt).

    Each line typically looks like: ``image_id variant name with spaces``.

    Returns
    -------
    list[tuple[str, str]]
        (image_id, class_label) tuples.
    """
    pairs: List[Tuple[str, str]] = []
    for line in txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        if len(parts) != 2:
            # Some files could be 'image_id class_label' but ensure robust fallback
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
        img_id, label = parts[0], parts[1]
        pairs.append((img_id, label))
    return pairs


def _safe_label(name: str) -> str:
    """Sanitize class label to be a safe folder name."""
    safe = "".join(c if c.isalnum() or c in ("-", "_")
                   else "-" for c in name.strip().lower())
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-")


def build_imagefolder_from_fgvc(
    fgvc_root: Path,
    split_out: Path,
    split: str = "trainval",
    copy: bool = True,
) -> Dict[str, int]:
    """Materialize an **ImageFolder** structure from FGVC Aircraft annotations.

    It reads the FGVC mapping file for the chosen split, then copies/moves images into
    ``split_out/<split>/<class_name>/image.jpg`` folders.

    Parameters
    ----------
    fgvc_root : Path
        Root directory containing the **extracted** FGVC tree (e.g., ``.../fgvc-aircraft-2013b``).
    split_out : Path
        Destination directory for the ImageFolder layout (e.g., ``data``).
    split : str, optional
        Which mapping to use: ``"trainval"`` or ``"test"``, by default ``"trainval"``.
    copy : bool, optional
        If True, copy images; if False, move them, by default True.

    Returns
    -------
    dict[str, int]
        Mapping from sanitized class name to index.
    """
    data_dir = fgvc_root / "data"
    images_dir = data_dir / "images"
    if split not in {"trainval", "test"}:
        raise ValueError("split must be 'trainval' or 'test'")
    mapping_file = data_dir / f"images_variant_{split}.txt"
    if not mapping_file.exists():
        raise FileNotFoundError(
            f"Expected mapping file not found: {mapping_file}")

    pairs = _read_mapping_file(mapping_file)
    # Build label set
    labels = sorted({_safe_label(lbl) for _, lbl in pairs})
    label2idx = {lbl: i for i, lbl in enumerate(labels)}

    target_base = split_out / split
    if target_base.exists():
        shutil.rmtree(target_base)
    for lbl in labels:
        (target_base / lbl).mkdir(parents=True, exist_ok=True)

    for img_id, lbl in pairs:
        src = images_dir / f"{img_id}.jpg"
        if not src.exists():
            # Some images might be .png, but FGVC ships .jpg typically
            continue
        safe_lbl = _safe_label(lbl)
        dst = target_base / safe_lbl / src.name
        if copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)

    return label2idx


def ensure_train_val_from_trainval(
    split_root: Path,
    out_root: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    copy: bool = True,
) -> None:
    """Create a stratified (per-class) train/val split from ``trainval`` folder.

    Parameters
    ----------
    split_root : Path
        Directory that contains ``trainval/<class_name>/*.jpg`` from FGVC mapping.
    out_root : Path
        Final output directory, containing ``train/`` and ``val/``.
    val_ratio : float
        Validation proportion per class (0-1), by default 0.2.
    seed : int
        Random seed for reproducibility.
    copy : bool
        If True copy files; else move.
    """
    import random
    rng = random.Random(seed)

    src_trainval = split_root / "trainval"
    dst_train = out_root / "train"
    dst_val = out_root / "val"
    if dst_train.exists():
        shutil.rmtree(dst_train)
    if dst_val.exists():
        shutil.rmtree(dst_val)

    for class_dir in sorted([d for d in src_trainval.iterdir() if d.is_dir()]):
        files = sorted([p for p in class_dir.iterdir() if p.is_file()])
        if not files:
            # empty class folder; skip
            continue
        n_val = max(1, int(len(files) * val_ratio))
        rng.shuffle(files)
        val_files = set(files[:n_val])
        train_files = [p for p in files if p not in val_files]

        (dst_train / class_dir.name).mkdir(parents=True, exist_ok=True)
        (dst_val / class_dir.name).mkdir(parents=True, exist_ok=True)
        for f in train_files:
            _copy_or_move(f, dst_train / class_dir.name / f.name, copy)
        for f in val_files:
            _copy_or_move(f, dst_val / class_dir.name / f.name, copy)


def _copy_or_move(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)
