from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import streamlit as st

class BaseClassifier(ABC):
    """Common interface for all classifiers."""

    @abstractmethod
    def build(self, input_shape: Tuple[int, ...], num_classes: int) -> None:
        """Create underlying model/graph given input shape and number of classes."""

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Train the model. Return a history/dict-like object."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class indices for a batch of samples."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for a batch of samples, shape (N, C)."""

    @abstractmethod
    def save(self, out_dir: Path) -> Path:
        """Persist model to `out_dir`. Return the path to the main artifact."""

    @classmethod
    @abstractmethod
    @st.cache_data
    def load(cls, path: Path) -> "BaseClassifier":
        """Load a persisted model from disk and return the instance."""
