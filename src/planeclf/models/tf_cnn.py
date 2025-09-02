from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

from .base import BaseClassifier

class TFCNNClassifier(BaseClassifier):
    """Simple Keras CNN matching your notebook structure."""

    def __init__(self):
        self.model: Optional[tf.keras.Model] = None
        # prevent TF from grabbing all VRAM
        try:
            for g in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    def build(self, input_shape: Tuple[int, ...], num_classes: int) -> None:
        model = Sequential()
        model.add(tf.keras.Input(shape=input_shape))
        model.add(Conv2D(32, (5, 5), activation="relu"))
        model.add(Conv2D(32, (5, 5), activation="relu"))
        model.add(MaxPool2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPool2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation="softmax"))
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.model = model

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        if self.model is None:
            self.build(X_train.shape[1:], y_train.shape[1])
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
            epochs=kwargs.get("epochs", 5),
            batch_size=kwargs.get("batch_size", 32),
            verbose=kwargs.get("verbose", 1),
            callbacks=kwargs.get("callbacks", []),
        )
        return history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.max() > 1.0:
            X = X.astype(np.float32) / 255.0
        return self.model.predict(X, verbose=0)  # type: ignore

    def save(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "tf_cnn.h5"
        self.model.save(path)  # type: ignore
        return path

    @classmethod
    def load(cls, path: Path) -> "TFCNNClassifier":
        obj = cls()
        obj.model = tf.keras.models.load_model(f"{path}.h5")  # type: ignore
        return obj
