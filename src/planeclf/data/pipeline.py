from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from planeclf.config import Config

def read_mapping_csv(csv_path: Path, target_name: str) -> pd.DataFrame:
    """Read a mapping CSV with lines 'image_id label with spaces' → DataFrame(image_id, path, target)."""
    df = pd.read_csv(csv_path, names=["raw"])
    config = Config.get()
    parts = df["raw"].str.split(" ", n=1, expand=True)
    df["image_id"] = parts[0]
    df[target_name] = parts[1].fillna("")
    df["path"] = df["image_id"].apply(lambda s: str(Path(config.paths.data_dir) / "images" / f"{s}.jpg"))
    return df[["image_id", "path", target_name]]

def load_and_resize(paths: List[str], size: Tuple[int, int]) -> np.ndarray:
    """Load list of image paths and resize to (H, W). Returns float32 in [0,1]."""
    H, W = size
    out = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((W, H))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        out.append(arr)
    return np.stack(out, axis=0)  # (N, H, W, 3)

def encode_labels(series: pd.Series) -> Tuple[np.ndarray, List[str]]:
    """Return one-hot matrix (N, C) and the class names."""
    cat = series.astype("category")
    names = list(cat.cat.categories)
    y_idx = cat.cat.codes.to_numpy()
    C = len(names)
    y = np.eye(C, dtype=np.float32)[y_idx]
    return y, names

def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """Stratified split using y-argmax as the label."""
    y_idx = np.argmax(y, axis=1)
    return train_test_split(X, y, test_size=val_ratio, random_state=seed, stratify=y_idx)
