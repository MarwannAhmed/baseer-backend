import pickle
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from arabic_ocr.config import TOP_K, MODELS_DIR
from arabic_ocr.segment.dots import Dot
from arabic_ocr.features import extract
from .base import BaseClassifier


class SVMClassifier(BaseClassifier):

    def __init__(self, C = 10.0, gamma = "scale"):
        self.svm     = SVC(C=C, kernel="rbf", gamma=gamma, probability=True)
        self.scaler  = StandardScaler()
        self.encoder = LabelEncoder()

    def train(self, X, y) -> None:
        self.encoder.fit(y)
        y_encoder = self.encoder.transform(y) #label encoder
        X_scaled = self.scaler.fit_transform(X)
        self.svm.fit(X_scaled, y_encoder)

    def predict(self, character_image, dot_list = None):
        feat = extract(character_image, dot_list).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)
        proba = self.svm.predict_proba(feat_scaled)[0]
        return self._top_k(proba)

    def predict_batch(self, character_images, dot_lists = None):
        if dot_lists is None:
            dot_lists = [None] * len(character_images)
        features = np.stack([extract(img, d) for img, d in zip(character_images, dot_lists)])
        features_scaled = self.scaler.transform(features)
        probas = self.svm.predict_proba(features_scaled)
        return [self._top_k(row) for row in probas]

    def _top_k(self, proba):
        k = min(TOP_K, len(proba))
        indices = np.argsort(proba)[::-1][:k]
        return [(str(self.encoder.classes_[i]), float(proba[i])) for i in indices]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"svm": self.svm, "scaler": self.scaler,
                         "encoder": self.encoder}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.svm = data["svm"]
        self.scaler = data["scaler"]
        self.encoder = data["encoder"]
