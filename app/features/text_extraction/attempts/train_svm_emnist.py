import os

import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from torchvision import datasets

from app.features.text_extraction.config import parameters
from app.features.text_extraction.utils.features import extract_batch

SPLIT         = "byclass"
MAX_PER_CLASS = 1000


def _load_emnist():
    train = datasets.EMNIST(root=parameters["train_dirs"]["emnist"], split=SPLIT, train=True,  download=True)
    test  = datasets.EMNIST(root=parameters["train_dirs"]["emnist"], split=SPLIT, train=False, download=True)

    imgs   = np.concatenate([train.data.numpy(),  test.data.numpy()],  axis=0)
    labels = np.concatenate([train.targets.numpy(), test.targets.numpy()], axis=0)
    mapping = dict(train.class_to_idx)
    idx_to_char = {v: k for k, v in mapping.items()}
    return imgs, labels, idx_to_char

def _map_to_lowercase(char):
    if char is None:
        return None
    if len(char) == 1 and char.isalpha():
        return char.lower()
    return char


def _prepare_samples(imgs, labels, idx_to_char):
    from collections import defaultdict
    import cv2

    buckets = defaultdict(list)
    for img, label in zip(imgs, labels):
        buckets[label].append(img)

    samples, targets = [], []
    for label, images in buckets.items():
        char = idx_to_char.get(int(label), None)
        if char is None:
            continue
        # char_lower = _map_to_lowercase(char)
        # if char_lower is None:
        #     continue
        subset = images[:MAX_PER_CLASS]
        for img in subset:
            arr = img.numpy() if hasattr(img, "numpy") else img
            arr = np.transpose(arr)
            # arr = np.fliplr(arr)
            arr = ((1.0 - arr / 255.0) * 255).astype(np.uint8)
            samples.append(arr)
            targets.append(char)

    return samples, targets


def train():
    os.makedirs(parameters["train_dirs"]["models"], exist_ok=True)
    os.makedirs(parameters["train_dirs"]["emnist"], exist_ok=True)

    print("Loading EMNIST...")
    imgs, labels, idx_to_char = _load_emnist()

    print("Preparing samples...")
    samples, targets = _prepare_samples(imgs, labels, idx_to_char)
    print(f"  {len(samples)} samples across {len(set(targets))} classes")
    # for i, (img, label) in enumerate(zip(samples[:35], targets[:35])):
    #     cv2.imwrite(f"output/sample_{i}_{label}.png", img)
    print("Extracting features...")
    X = extract_batch(samples)

    encoder = LabelEncoder()
    y       = encoder.fit_transform(targets)

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training SVM...")
    clf = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        decision_function_shape="ovr",
        class_weight="balanced",
    )
    clf.fit(X_scaled, y)

    scores = cross_val_score(clf, X_scaled, y, cv=3, scoring="accuracy", n_jobs=-1)
    print(f"Cross-val accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    with open(parameters["SVM"]["model_path"],   "wb") as f: pickle.dump(clf,     f)
    with open(parameters["SVM"]["scaler_path"],  "wb") as f: pickle.dump(scaler,  f)
    with open(parameters["SVM"]["encoder_path"], "wb") as f: pickle.dump(encoder, f)
    print(f"Saved: {parameters['SVM']['model_path']}, {parameters['SVM']['scaler_path']}, {parameters['SVM']['encoder_path']}")


if __name__ == "__main__":
    train()