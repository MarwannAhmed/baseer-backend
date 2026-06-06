import joblib
import numpy as np

from app.features.text_extraction.utils.features import extract
from app.features.text_extraction.config import parameters

class SVMClassifier:
    def __init__(self):
        self._clf     = joblib.load(parameters["SVM"]["model_path"])
        self._scaler  = joblib.load(parameters["SVM"]["scaler_path"])
        self._encoder = joblib.load(parameters["SVM"]["encoder_path"])

    def predict(self, char_img):
        feat     = extract(char_img).reshape(1, -1)
        scaled   = self._scaler.transform(feat)
        probs    = self._clf.predict_proba(scaled)[0]
        top_idx  = np.argsort(probs)[::-1][:parameters["SVM"]["TOP_K"]]
        return [
            (self._encoder.inverse_transform([i])[0], float(probs[i]))
            for i in top_idx
        ]

    def predict_best(self, char_img):
        candidates = self.predict(char_img)
        return candidates[0]

    def predict_batch(self, char_imgs):
        from app.features.text_extraction.utils.features import extract_batch
        feats   = extract_batch(char_imgs)
        scaled  = self._scaler.transform(feats)
        probs   = self._clf.predict_proba(scaled)
        results = []
        for row in probs:
            top_idx = np.argsort(row)[::-1][:parameters["SVM"]["TOP_K"]]
            results.append([
                (self._encoder.inverse_transform([i])[0], float(row[i]))
                for i in top_idx
            ])
        return results