import pickle
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from arabic_ocr.config import MODELS_DIR
from arabic_ocr.utils import arabic_utils as _au


class LearnedReranker:

    def __init__(self):
        self.model_path = Path(MODELS_DIR) / "reranker.pkl"
        self.model = None
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
            except Exception:
                self.model = None

    def rerank(self, candidates, observed_dot_feat = None):
        if not candidates:
            return candidates

        if observed_dot_feat is None:
            obs_above = 0.0
            obs_below = 0.0
            obs_has = 0.0
            obs_spread = 0.0
        else:
            obs_above = float(observed_dot_feat[0])
            obs_below = float(observed_dot_feat[1])
            obs_has = float(observed_dot_feat[2])
            obs_spread = float(observed_dot_feat[3])

        if self.model is None:
            boost = 0.20
            penalty = 0.10
            reranked = []
            for lab, conf in candidates:
                base = lab.rsplit("_", 1)[0]
                expected = _au._LETTER_DOT_COUNTS.get(base)
                new_conf = conf
                if expected is not None:
                    if expected == (int(obs_above), int(obs_below)):
                        new_conf = conf + boost
                    else:
                        new_conf = max(0.0, conf - penalty)
                reranked.append((lab, new_conf))
            reranked.sort(key=lambda t: t[1], reverse=True)
            total = sum(c for _, c in reranked) or 1.0
            return [(lab, c / total) for lab, c in reranked]

        X = []
        for lab, conf in candidates:
            base = lab.rsplit("_", 1)[0]
            exp = _au._LETTER_DOT_COUNTS.get(base, (0, 0))
            X.append([float(conf), float(exp[0]), float(exp[1]), obs_above, obs_below, obs_has, obs_spread])
        X = np.asarray(X, dtype=np.float32)
        try:
            probs = self.model.predict_proba(X)[:, 1]
        except Exception:
            probs = self.model.predict(X)

        #combines model score with original confidence (weighted average)
        if probs is None:
            reranked = candidates
        else:
            alpha = 0.6  # weight for original classifier confidence
            reranked = []
            for (lab, conf), p in zip(candidates, probs):
                combined = float(alpha * conf + (1.0 - alpha) * float(p))
                reranked.append((lab, combined))

        total = sum(c for _, c in reranked) or 1.0
        return [(lab, c / total) for lab, c in reranked]

RERANKER = LearnedReranker()


def save_model(model, path = None):
    p = Path(path) if path is not None else Path(MODELS_DIR) / "reranker.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(model, f)
