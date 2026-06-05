import json
import os
import numpy as np
import cv2
import joblib

from app.features.object_detection import config
from app.features.object_detection.features import extract_features
from app.features.object_detection.utils import non_maximum_suppression, compute_iou
from app.core.image_utils import decode_image, resize_if_large

def load_models():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    models_folder = os.path.join(root, "models/object")
    class_labels = ["person", "bottle", "chair", "dining table", "handbag"]

    models = {}
    for label in class_labels:
        path = os.path.join(models_folder, f"detector_{label}.pkl")
        if not os.path.isfile(path):
            print(f"No pickel file found for '{label}' with path ({path}).")
            continue
        models[label] = joblib.load(path)
        print(f"Loaded {label} class detector.")

    return models, class_labels

models, class_labels = load_models()

def detect_objects(image, confidence_threshold=0.85):
    name_index_map = {name: index for index, name in enumerate(class_labels)}
    
    # region proposals using selective search
    selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selective_search.setBaseImage(image)
    selective_search.switchToSelectiveSearchFast()
    raw = selective_search.process()

    region_proposals = []
    for (x_coordinate, y_coordinate, width, height) in raw[:500]:
        if width < 48 or height < 48:
            continue
        x1, y1, x2, y2 = x_coordinate, y_coordinate, x_coordinate + width, y_coordinate + height
        region_proposals.append([x1, y1, x2, y2])

    print(f" Selective Search results: {len(raw)} raw proposals, reduced to "f"{len(region_proposals)} after filtering.")

    all_boxes  = []
    all_scores = []
    all_labels = []
    for (x1, y1, x2, y2) in region_proposals:
        patch = image[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        try:
            feature_vector = extract_features(patch).reshape(1, -1)
        except Exception:
            continue

        for class_label, model in models.items():
            probability = model.predict_proba(feature_vector)[0][1]
            if probability >= confidence_threshold:
                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(float(probability))
                all_labels.append(name_index_map[class_label])

    boxes, scores, labels = non_maximum_suppression(all_boxes, all_scores, all_labels, iou_threshold=0.15)
    if len(boxes) > 1:
        boxes_array  = np.array(boxes,  dtype=np.float32)
        scores_array = np.array(scores, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int32)

        ordered_scores = np.argsort(scores_array)[::-1]
        keep  = []
        while len(ordered_scores) > 0:
            best = ordered_scores[0]
            keep.append(best)
            if len(ordered_scores) == 1:
                break
            rest = ordered_scores[1:]
            ious = np.array([compute_iou(boxes_array[best], boxes_array[i]) for i in rest])
            ordered_scores = rest[ious < 0.15]

        boxes  = boxes_array[keep].tolist()
        scores = scores_array[keep].tolist()
        labels = labels_array[keep].tolist()

    return boxes, scores, labels

def run_detection(image_bytes: bytes) -> list:
    image = decode_image(image_bytes)
    if image is None:
        return []
    image = resize_if_large(image)

    try:
        boxes, scores, labels = detect_objects(image)
    except Exception as error:
        print(f"Error during object detection: {error}.")
        return []

    results = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        results.append({
            "label": class_labels[label],
            "confidence": round(score, 4),
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
        })

    return results