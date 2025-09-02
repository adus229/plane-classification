from __future__ import annotations
from typing import Type, Dict

from .tf_imagenet import TFImaNetClassifier
from .base import BaseClassifier
from .svm_hog import SVMHOGClassifier
from .tf_cnn import TFCNNClassifier

_REGISTRY: Dict[str, Type[BaseClassifier]] = {
    "svm_hog": {
        "class": SVMHOGClassifier,
        "label": "SVM HOG"
    },
    "tf_cnn": {
        "class": TFCNNClassifier,
        "label": "CNN"
    },

    "tf_imagenet": {
        "class": TFImaNetClassifier,
        "label": "TF ImageNet"
    },   # ← ajout
    # "tf_transfer": TFTransferClassifier,  # à ajouter plus tard
}

available_models = list(sorted(_REGISTRY.keys()))


def get_model_class(name: str) -> Type[BaseClassifier]:
    name = name.lower()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]["class"]

def get_model_label(name: str) -> str:
    name = name.lower()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]["label"]
