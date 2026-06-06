import os
import json
import random

import numpy as np
import cv2
import joblib
from tqdm import tqdm

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle as sk_shuffle

from app.features.object_detection.features import extract_features

random.seed(42)
np.random.seed(42)


def load_data():
    positives = {}
    class_labels = ["person", "bottle", "chair", "dining table", "handbag"]
    for class_label in class_labels:
        path = os.path.join("data", f"positives_{class_label}.npy")
        if not os.path.isfile(path):
            print(f"NumPy files not prepared for class {class_label}")
            return None, None
        class_positives = np.load(path)
        positives[class_label] = class_positives
        print(f"For {class_label} loaded {len(class_positives)} positives.")

    negatives_path = os.path.join("data", "negatives.npy")
    if not os.path.isfile(negatives_path):
        print(f"NumPy file not prepared for negative data.")
        return None, None
    negatives = np.load(negatives_path)
    print(f"For backgorund loaded {len(negatives)} negatives.")

    return positives, negatives


def build_pipeline():
    linear_svc = LinearSVC(C=0.01, max_iter=10_000)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", CalibratedClassifierCV(linear_svc, cv=3)),
        ]
    )


def train_class(class_name, positive_features, negative_features):
    positive_labels = np.ones(len(positive_features), dtype=np.int32)
    negative_labels = np.zeros(len(negative_features), dtype=np.int32)

    training_features = np.vstack([positive_features, negative_features])
    training_labels = np.concatenate([positive_labels, negative_labels])

    training_features, training_labels = sk_shuffle(
        training_features, training_labels, random_state=42
    )

    model = build_pipeline()
    model.fit(training_features, training_labels)

    training_accuracy = model.score(training_features, training_labels)

    print(f"Training accuracy for {class_name}: {training_accuracy:.4f}")

    return model


def mine_hard_negatives(model, image_list, class_name):
    hard_negatives = []
    window_width, window_height = (96, 96)
    stride = 32

    for image_path in tqdm(image_list[:150], desc="mining", leave=False):
        image = cv2.imread(image_path)
        if image is None:
            continue

        height, width = image.shape[:2]
        if width < window_width or height < window_height:
            continue

        for row in range(0, height - window_height + 1, stride):
            for column in range(0, width - window_width + 1, stride):
                patch = image[row : row + window_height, column : column + window_width]
                try:
                    feature_vector = extract_features(patch).reshape(1, -1)
                    probability = model.predict_proba(feature_vector)[0][1]
                    if probability >= 0.5:
                        hard_negatives.append(feature_vector.flatten())
                except Exception:
                    continue

                if len(hard_negatives) >= 1500:
                    break
            if len(hard_negatives) >= 1500:
                break
        if len(hard_negatives) >= 1500:
            break

    return np.array(hard_negatives, dtype=np.float32) if hard_negatives else None


def main():
    os.makedirs("models", exist_ok=True)

    positives, negatives = load_data()
    if positives is None:
        return

    path = os.path.join("data", "image_list.json")
    if not os.path.isfile(path):
        image_list = []
    with open(path) as file:
        image_list = json.load(file)

    models = {}

    class_labels = ["person", "bottle", "chair", "dining table", "handbag"]
    for class_label in class_labels:
        class_positives = positives[class_label]
        if len(class_positives) == 0:
            print(f"No positive samples found for {class_label} class.")
            continue

        model = train_class(class_label, class_positives, negatives)
        models[class_label] = model

    if not models:
        return

    if image_list:
        for class_label in list(models.keys()):
            hard_negatives = mine_hard_negatives(
                models[class_label], image_list, class_label
            )

            if hard_negatives is not None and len(hard_negatives) > 0:
                augmented_negatives = np.vstack([negatives, hard_negatives])
                models[class_label] = train_class(
                    class_label, positives[class_label], augmented_negatives
                )
    else:
        print("No image list for hard negative mining.")

    for class_label, model in models.items():
        path = os.path.join("models", f"detector_{class_label}.pkl")
        joblib.dump(model, path)

    classes_path = os.path.join("models", "classes.json")
    with open(classes_path, "w") as f:
        json.dump(list(models.keys()), f)


if __name__ == "__main__":
    main()
