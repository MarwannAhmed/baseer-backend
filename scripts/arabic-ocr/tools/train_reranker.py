import argparse
import pickle
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from app.features.arabic_ocr.utils.image_io import load_image
from app.features.arabic_ocr.classifiers import get_classifier
from app.features.arabic_ocr.config import DATA_DIR, MODELS_DIR, TOP_K
from app.features.arabic_ocr.utils import arabic_utils as _au

def iter_char_images(data_dir):
    chars_dir = data_dir / "chars"
    if not chars_dir.exists():
        return
    for label_dir in sorted(chars_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for img_path in sorted(label_dir.glob("*.png")):
            yield label, img_path


def build_dataset(classifier_name, negatives_per_sample = 4):
    labels = sorted({lab for lab, _ in iter_char_images(Path(DATA_DIR))})
    if not labels:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)

    X = []
    y = []
    import random

    # Try to use classifier top-K outputs for negative candidates when available
    clf = get_classifier(classifier_name)
    for true_label, img_path in iter_char_images(Path(DATA_DIR)):
        base = true_label.rsplit("_", 1)[0]
        exp = _au._LETTER_DOT_COUNTS.get(base, (0, 0))
        obs_above, obs_below = exp
        obs_has = 1.0 if (obs_above + obs_below) > 0 else 0.0
        obs_spread = 0.0
        X.append([1.0, float(exp[0]), float(exp[1]), float(obs_above), float(obs_below), obs_has, float(obs_spread)])
        y.append(1)

        try:
            cands = clf.predict(img_path if hasattr(clf, 'predict') else None, None)
        except Exception:
            cands = []

        negs = []
        if cands:
            for cand_label, conf in cands:
                if cand_label == true_label:
                    continue
                base_c = cand_label.rsplit("_", 1)[0]
                exp_c = _au._LETTER_DOT_COUNTS.get(base_c, (0, 0))
                negs.append((conf, exp_c))
        neg_choices = [l for l in labels if l != true_label]
        while len(negs) < negatives_per_sample and neg_choices:
            neg = random.choice(neg_choices)
            conf = float(random.uniform(0.0, 0.9))
            nb = neg.rsplit("_", 1)[0]
            nexp = _au._LETTER_DOT_COUNTS.get(nb, (0, 0))
            negs.append((conf, nexp))

        for conf, nexp in negs[:negatives_per_sample]:
            X.append([conf, float(nexp[0]), float(nexp[1]), float(obs_above), float(obs_below), obs_has, 0.0])
            y.append(0)

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classifier", default="svm")
    args = p.parse_args()

    print("Building dataset from data/chars (this may take a while)...")
    X, y = build_dataset(args.classifier)
    print(f"Collected {len(y)} training rows")
    if len(y) < 50:
        print("Not enough data to train; need more samples")
        return

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(MODELS_DIR) / "langmodel-ocr-ar" / "reranker.pkl"
    with open(out, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved trained reranker to {out}")


if __name__ == "__main__":
    main()
