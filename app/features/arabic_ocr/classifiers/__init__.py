from pathlib import Path
from arabic_ocr.config import MODELS_DIR
from .base import BaseClassifier
from .svm_classifier import SVMClassifier
from .rf_classifier import RFClassifier
from .cnn_classifier import CNNClassifier

_REGISTRY = {
    "cnn": (CNNClassifier, MODELS_DIR / "cnn-ocr-ar" / "model.pt"),
}

def get_classifier(name: str) -> BaseClassifier:
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown classifier '{name}'. Choose from: {list(_REGISTRY)}")

    cls, default_path = _REGISTRY[name]
    clf = cls()
    if Path(default_path).exists():
        clf.load(default_path)
    return clf


__all__ = ["BaseClassifier", "SVMClassifier", "RFClassifier", "CNNClassifier","get_classifier"]
