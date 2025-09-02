from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras import layers as KL
from tensorflow.keras import mixed_precision

from .base import BaseClassifier


class TFImaNetClassifier(BaseClassifier):
    """Transfer learning classifier using an ImageNet-pretrained backbone (Xception by default).

    Parameters
    ----------
    base : str
        Backbone name. Currently supports: "Xception". (Extendable later.)
    freeze_base : bool
        If True, keep the backbone frozen (feature extractor). If False, train the whole network.
    fine_tune_at : int | None
        If set, unfreezes the backbone and makes all layers from this index trainable (partial fine-tuning).
        Ignored if `freeze_base=True`.
    dropout : float
        Dropout rate before the final Dense head.
    policy : str
        "float32" or "mixed_float16". Using mixed precision reduces memory use on RTX GPUs.
    """

    def __init__(
        self,
        base: str = "Xception",
        freeze_base: bool = True,
        fine_tune_at: Optional[int] = None,
        dropout: float = 0.2,
        policy: str = "float32",
    ):
        self.base = base
        self.freeze_base = bool(freeze_base)
        self.fine_tune_at = fine_tune_at
        self.dropout = float(dropout)
        self.policy = str(policy)

        self.model: Optional[Model] = None

        # Be nice with GPU memory
        try:
            for g in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    # ---- BaseClassifier API ----

    def build(self, input_shape: Tuple[int, ...], num_classes: int) -> None:
        """Create and compile the transfer model."""
        # clear previous graph
        tf.keras.backend.clear_session()

        # precision policy
        try:
            mixed_precision.set_global_policy("mixed_float16" if self.policy in (
                "mixed_float16", "float16", "fp16") else "float32")
        except Exception:
            pass

        # --- Backbone (ImageNet, no top) ---
        if self.base.lower() == "xception":
            backbone = tf.keras.applications.Xception(
                weights="imagenet",
                include_top=False,
                input_shape=input_shape,
            )
        else:
            raise ValueError(
                f"Unsupported backbone '{self.base}'. Supported: Xception")

        # freeze / partial fine-tune
        if self.freeze_base:
            backbone.trainable = False
        else:
            backbone.trainable = True
            if isinstance(self.fine_tune_at, int):
                for i, layer in enumerate(backbone.layers):
                    layer.trainable = i >= self.fine_tune_at

        # --- Input & augmentation ---
        inputs = tf.keras.Input(shape=input_shape)
        aug = Sequential(
            [
                KL.RandomFlip("horizontal"),
                KL.RandomRotation(0.1),
            ],
            name="augmentation",
        )
        x = aug(inputs)

        # Xception expects [-1, 1] scaled inputs
        x = KL.Rescaling(scale=1 / 127.5, offset=-1.0, name="rescale_m1_1")(x)

        # Important: if the backbone is frozen, force it to run in inference mode
        # so BatchNorm layers don't update moving stats.
        x = backbone(
            x, training=not self.freeze_base and self.fine_tune_at is not None)

        # --- Head ---
        x = KL.GlobalAveragePooling2D(name="gap")(x)
        x = KL.Dropout(self.dropout, name="dropout")(x)
        # Keep classifier head in float32 for numerical stability under mixed precision
        outputs = KL.Dense(num_classes, activation="softmax",
                           dtype="float32", name="probs")(x)

        model = tf.keras.Model(
            inputs, outputs, name=f"tf_imagenet_{self.base.lower()}")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy",
                     tf.keras.metrics.CategoricalCrossentropy(name="cce")],
        )
        self.model = model

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        if self.model is None:
            self.build(X_train.shape[1:], y_train.shape[1])
        history = self.model.fit(  # type: ignore
            X_train,
            y_train,
            validation_data=(
                X_val, y_val) if X_val is not None and y_val is not None else None,
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
            # our head expects [-1,1] preprocessing; apply same rescaling used in the model
            X = X * 2.0 - 1.0
        return self.model.predict(X, verbose=0)  # type: ignore

    def save(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "tf_imagenet.h5"
        self.model.save(path)  # type: ignore
        return path

    @classmethod
    def load(cls, path: Path) -> "TFImaNetClassifier":
        obj = cls()
        obj.model = tf.keras.models.load_model(f"{path}.h5")  # type: ignore
        return obj
