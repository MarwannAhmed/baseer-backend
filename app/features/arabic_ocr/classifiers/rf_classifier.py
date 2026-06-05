import pickle
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from arabic_ocr.config import TOP_K
from arabic_ocr.segment.dots import Dot
from arabic_ocr.features import extract
from .base import BaseClassifier


class RFClassifier(BaseClassifier):

    def __init__(self, n_estimators: int = 300, n_jobs: int = -1):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=None,
            n_jobs=n_jobs, 
            random_state=42, 
        )
        self.encoder = LabelEncoder()

    def train(self, X, y):
        self.encoder.fit(y)
        self.rf.fit(X, self.encoder.transform(y))

    def predict(self, character_image, dot_list = None):
        feat = extract(character_image, dot_list).reshape(1, -1)
        proba = self.rf.predict_proba(feat)[0]
        return self._top_k(proba)

    def predict_batch(self, character_images, dot_lists = None):
        if dot_lists is None:
            dot_lists = [None] * len(character_images)
        feats = np.stack([extract(img, d) for img, d in zip(character_images, dot_lists)])
        probas = self.rf.predict_proba(feats)
        return [self._top_k(row) for row in probas]

    def _top_k(self, proba):
        k = min(TOP_K, len(proba))
        indices = np.argsort(proba)[::-1][:k]
        return [(str(self.encoder.classes_[i]), float(proba[i])) for i in indices]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"rf": self.rf, "encoder": self.encoder}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.rf = data["rf"]
        self.encoder = data["encoder"]
