from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List
import shutil

import numpy as np
import yaml

from planeclf.config import Config

from planeclf.models.registry import available_models
from ..data.pipeline import (
    read_mapping_csv,
    load_and_resize,
    encode_labels,
    stratified_split,
)
from ..models.registry import get_model_class


def _parse_models_arg(arg: str, available: Iterable[str]) -> List[str]:
    """Accept 'all' or comma-separated list."""
    arg = (arg or "").strip().lower()
    if arg in ("all", "*"):
        return list(sorted(set(available)))
    items = [s.strip().lower() for s in arg.split(",") if s.strip()]
    if not items:
        raise ValueError("Empty --models argument")
    return items


def _save_classes(classes: List[str], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "classes.yaml"
    path.write_text(yaml.safe_dump(list(classes)))
    return path


def _compute_accuracy(y_true_onehot: np.ndarray, y_pred_idx: np.ndarray) -> float:
    y_true_idx = np.argmax(y_true_onehot, axis=1)
    return float(np.mean(y_true_idx == y_pred_idx))


def _safe_swap_dir(tmp_dir: Path, slot_dir: Path) -> None:
    """Atomically replace slot_dir with tmp_dir."""
    if slot_dir.exists():
        shutil.rmtree(slot_dir)
    tmp_dir.rename(slot_dir)


def _canonicalize_artifact(tmp_artifact: Path, model_name: str) -> Path:
    """
    Rename the saved artifact to <model_name>.<ext> inside its directory.
    Returns the canonical path.
    """
    ext = "".join(tmp_artifact.suffixes) or ""  # ex: ".h5", ".joblib"
    canonical = tmp_artifact.parent / f"{model_name}{ext}"
    if tmp_artifact != canonical:
        tmp_artifact.replace(canonical)
    return canonical


def train_one_model(
    model_name: str,
    target: str,
    val_ratio: float = 0.2,
    batch_size: int = 32,
    epochs: int = 5,
    params: Dict[str, Any] | None = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Entraîne UN modèle et sauvegarde en **slot unique**:
    models/<model_name>/{<model_name>.<ext>, classes.yaml, metrics.json}
    """
    config = Config.get()

    train_paths = config. paths.get_train_paths(target)
    train_csv: Path = train_paths["train"]
    val_csv: Path = train_paths["val"]
    size = (config.img_width, config.img_height)

    # ---- Data (indépendant des modèles) ----
    train_df = read_mapping_csv(train_csv,target)
    X_all = load_and_resize(train_df["path"].tolist(), size)
    y_all, class_names = encode_labels(train_df[target])

    if val_csv:
        val_df = read_mapping_csv(val_csv, target)
        X_val = load_and_resize(val_df["path"].tolist(), size)
        y_val, _ = encode_labels(val_df[target])
        X_train, y_train = X_all, y_all
    else:
        X_train, X_val, y_train, y_val = stratified_split(
            X_all, y_all, val_ratio=val_ratio, seed=seed)

    # ---- Build & train ----
    ModelClass = get_model_class(model_name)

    model =  ModelClass()

    model.build(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])

# Train the model
    history = model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        batch_size=batch_size, epochs=epochs, verbose=1
    )

    # Compute the accuracy on the validation set
    y_pred_idx = model.predict(X_val)
    acc = _compute_accuracy(y_val, y_pred_idx)

    # ---- Sauvegarde "slot unique" (swap sûr) ----
    slot_dir = config.paths.models_dir /target/ model_name
    tmp_dir = config.paths.models_dir /target/ f".tmp_{model_name}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1) on sauve DANS le tmp
    tmp_artifact = model.save(tmp_dir)
    artifact_path = _canonicalize_artifact(
        tmp_artifact, model_name)  # tf_cnn.h5 → tf_cnn.h5 (ou rename)
    classes_path = _save_classes(class_names, tmp_dir)

    # 2) metrics.json (toujours écrasé)
    # On capture la "fin" d'historique si dispo
    last_history: Dict[str, Any] = {}
    if isinstance(history, dict):
        last_history = {
            f"final_{k}": (float(v[-1]) if isinstance(v, (list, tuple)) else float(v))
            for k, v in history.items()
        }

    summary = {
        "model": model_name,
        "model_path": str(artifact_path),
        "classes_path": str(classes_path),
        "num_classes": int(y_train.shape[1]),
        "samples": {"train": int(len(X_train)), "val": int(len(X_val))},
        "val_accuracy": acc,
        "constructor_params": params,
        "history_tail": last_history,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (tmp_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    # 3) swap atomique: remplace l'ancien slot
    _safe_swap_dir(tmp_dir, slot_dir)

    # Renvoie les chemins finals (dans slot_dir)
    return {
        "model": model_name,
        "model_path": str(slot_dir / Path(artifact_path).name),
        "classes_path": str(slot_dir / "classes.yaml"),
        "metrics_path": str(slot_dir / "metrics.json"),
        "val_accuracy": acc,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generic trainer to run one or all registered models.")
    # What to train
    p.add_argument("--models", required=True,
                   help="'tf_cnn', 'svm_hog', 'name1,name2', or 'all'")
    p.add_argument("--list", action="store_true",
                   help="List available model names and exit")
    p.add_argument("--target", required=True,
                   help="Name of target column (e.g., 'manufacturer')")

    args = p.parse_args()

    # Listing shortcut
    if args.list:
        # Discover from registry by trying to ask for a bogus name → get choices from error
        print("Available models:", ", ".join(available_models))
        return

    available = available_models # keep in sync with registry.py
    model_names = _parse_models_arg(args.models, available)

    summaries: List[Dict[str, Any]] = []
    for name in model_names:
        if name not in available:
            raise SystemExit(f"Unknown model '{name}'. Available: {available}")

        print(f"\n=== Training {name} ===")
        s = train_one_model(
            model_name=name,
            target=args.target,
        )
        print(json.dumps(
            {"model": name, "val_accuracy": s["val_accuracy"], "out": s["model_path"]}, indent=2))
        summaries.append(s)


if __name__ == "__main__":
    main()
