import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from app.features.arabic_ocr.config import MODELS_DIR, DATA_DIR
from app.features.arabic_ocr.classifiers import SVMClassifier
from app.features.arabic_ocr.features import extract
from app.features.arabic_ocr.features.normalize import normalize
from app.features.arabic_ocr.utils.arabic_utils import HMDB_TO_UNICODE, hmdb_label_to_unicode

try:
    import cv2
except ImportError as e:
    raise SystemExit("opencv-python is required: pip install opencv-python") from e


def load_dataset(data_dir):
    imgs, labels = [], []

    folders = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if not folders:
        raise FileNotFoundError(f"No subdirectories found in {data_dir}")

    for folder in folders:
        label = folder.name
        loaded = 0
        for img_path in sorted(folder.glob("*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            imgs.append(img)
            labels.append(label)
            loaded += 1

        if loaded:
            unicode_repr = hmdb_label_to_unicode(label)
            display = f" ({unicode_repr})" if unicode_repr else ""
            print(f"  {label}{display}: {loaded} images")

    if not imgs:
        raise RuntimeError(f"No PNG images found under {data_dir}")

    print(f"\nTotal: {len(imgs)} images across {len(set(labels))} classes")
    print("Extracting features …")

    X = np.stack([extract(img) for img in imgs])
    y = np.array(labels)
    return X, y


def main():
    parser = argparse.ArgumentParser(
        description="Train SVM on HMDB-style LetterName_Position folders"
    )
    parser.add_argument(
        "--data-dir", default=str(DATA_DIR / "chars"),
        help="Root directory containing LetterName_Position subfolders",
    )
    parser.add_argument(
        "--test-split", type=float, default=0.15,
        help="Fraction of data held out for evaluation (default 0.15)",
    )
    parser.add_argument(
        "--C", type=float, default=10.0,
        help="SVM regularisation parameter C (default 10.0)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    X, y = load_dataset(data_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=42, stratify=y,
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    clf = SVMClassifier(C=args.C)
    print("Fitting SVM (this may take a few minutes for large datasets) …")
    clf.train(X_train, y_train)

    X_test_scaled = clf.scaler.transform(X_test)
    proba = clf.svm.predict_proba(X_test_scaled)
    y_pred_labels = clf.encoder.inverse_transform(proba.argmax(axis=1))

    print("\n── Classification report ─────────────────────────────────────────")
    target_names = [
        f"{lbl} {hmdb_label_to_unicode(lbl)}".strip()
        for lbl in sorted(set(y_test))
    ]
    print(classification_report(
        y_test, y_pred_labels,
        labels=sorted(set(y_test)),
        target_names=target_names,
        zero_division=0,
    ))

    clf.save(Path(args.out))
    print(f"Model saved → {args.out}")


if __name__ == "__main__":
    main()
