from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .base import BaseClassifier


@dataclass
class HOGConfig:
    pixels_per_cell: Tuple[int, int] = (8, 8)
    cells_per_block: Tuple[int, int] = (2, 2)
    orientations: int = 9


class SVMHOGClassifier(BaseClassifier):
    """SVM classifier with HOG features."""

    def __init__(self, hog_cfg: Optional[HOGConfig] = None, C: float = 1.0, gamma: str | float = "scale"):
        self.hog_cfg = hog_cfg or HOGConfig()
        self.C = C
        self.gamma = gamma
        self.pipeline: Optional[Pipeline] = None
        self.num_classes: Optional[int] = None

    # --- private ---
    def _hog_batch(self, X: np.ndarray) -> np.ndarray:
        feats = []
        for img in X:
            # Expect img shape (H, W, 3); convert to grayscale for HOG
            gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
            f = hog(
                gray,
                orientations=self.hog_cfg.orientations,
                pixels_per_cell=self.hog_cfg.pixels_per_cell,
                cells_per_block=self.hog_cfg.cells_per_block,
                feature_vector=True,
            )
            feats.append(f)
        return np.asarray(feats, dtype=np.float32)

    # --- BaseClassifier API ---+
    def build(self, input_shape: Tuple[int, ...], num_classes: int) -> None:
        self.num_classes = num_classes
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                ("svc", SVC(C=self.C, gamma=self.gamma, probability=True)),
            ]
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, **kwargs):
        if self.pipeline is None:
            self.build(X_train.shape[1:], y_train.shape[1])
        Xf = self._hog_batch(X_train)
        # For scikit-learn we need class indices
        y_idx = np.argmax(y_train, axis=1)
        self.pipeline.fit(Xf, y_idx)
        return {"train_samples": int(len(X_train))}

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xf = self._hog_batch(X)
        idx = self.pipeline.predict(Xf)  # type: ignore
        return idx

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xf = self._hog_batch(X)
        return self.pipeline.predict_proba(Xf)  # type: ignore

    def save(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir /"svm_hog.joblib"
        joblib.dump({"pipeline": self.pipeline,
                    "num_classes": self.num_classes, "hog_cfg": self.hog_cfg}, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "SVMHOGClassifier":
        blob = joblib.load(f"{path}.joblib")
        obj = cls(hog_cfg=blob["hog_cfg"])
        obj.pipeline = blob["pipeline"]
        obj.num_classes = blob["num_classes"]
        return obj
