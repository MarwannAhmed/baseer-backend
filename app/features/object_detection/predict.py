"""
app/features/object_detection/predict.py — Inference
======================================================
Same detection logic as step3_predict.py.
Models are loaded once at server startup.
run_detection() is the only function called externally.
"""

import json
import os
import cv2

import numpy as np
import joblib

from app.features.object_detection import config
from app.features.object_detection.features import extract_features
from app.features.object_detection.utils import non_maximum_suppression, compute_iou
from app.core.image_utils import decode_image, resize_if_large


# ── Load models once at startup ───────────────────────────────────────────────

def _load_models():
    classes_path = os.path.join(config.MODELS_DIR, "classes.json")
    if os.path.isfile(classes_path):
        with open(classes_path) as f:
            class_names = json.load(f)
    else:
        class_names = config.CLASSES

    models = {}
    for cls in class_names:
        path = os.path.join(config.MODELS_DIR, f"detector_{cls}.pkl")
        if not os.path.isfile(path):
            print(f"  WARNING: No model found for '{cls}'  ({path})")
            continue
        models[cls] = joblib.load(path)
        print(f"  Loaded  [{cls}]")

    return models, class_names


_models, _class_names = _load_models()


# ── Detection logic (same as step3_predict.py detect()) ──────────────────────

def _detect(image, confidence_threshold=None):
    if confidence_threshold is None:
        confidence_threshold = config.CONFIDENCE_THRESHOLD

    name_to_idx = {name: i for i, name in enumerate(_class_names)}

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    raw = ss.process()

    proposals = []
    for (x, y, w, h) in raw[:config.SS_MAX_PROPOSALS]:
        if w < config.SS_MIN_BOX_SIZE or h < config.SS_MIN_BOX_SIZE:
            continue
        proposals.append([x, y, x + w, y + h])

    print(f"  Selective Search: {len(raw)} raw proposals → "
          f"{len(proposals)} after size filter")

    all_boxes  = []
    all_scores = []
    all_labels = []

    for (x1, y1, x2, y2) in proposals:
        patch = image[y1:y2, x1:x2]
        if patch.size == 0:
            continue

        try:
            feat = extract_features(patch).reshape(1, -1)
        except Exception:
            continue

        for cls_name, model in _models.items():
            prob = model.predict_proba(feat)[0][1]
            if prob >= confidence_threshold:
                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(float(prob))
                all_labels.append(name_to_idx[cls_name])

    boxes, scores, labels = non_maximum_suppression(
        all_boxes, all_scores, all_labels,
        iou_threshold=config.NMS_OVERLAP_THRESHOLD,
    )

    # Winner-takes-all across classes
    if len(boxes) > 1:
        boxes_arr  = np.array(boxes,  dtype=np.float32)
        scores_arr = np.array(scores, dtype=np.float32)
        labels_arr = np.array(labels, dtype=np.int32)

        order = np.argsort(scores_arr)[::-1]
        keep  = []
        while len(order) > 0:
            best = order[0]
            keep.append(best)
            if len(order) == 1:
                break
            rest = order[1:]
            ious = np.array([
                compute_iou(boxes_arr[best], boxes_arr[j]) for j in rest
            ])
            order = rest[ious < config.NMS_OVERLAP_THRESHOLD]

        boxes  = boxes_arr[keep].tolist()
        scores = scores_arr[keep].tolist()
        labels = labels_arr[keep].tolist()

    return boxes, scores, labels


# ── Public function ───────────────────────────────────────────────────────────

def run_detection(image_bytes: bytes) -> list:
    """
    Returns a list of detection dicts, each with:
      label      : str    class name
      score      : float  confidence 0.0–1.0
      top_right  : {x, y} top-right corner of bounding box
      bottom_left: {x, y} bottom-left corner of bounding box
    Returns an empty list if nothing detected or on error.
    """
    image = decode_image(image_bytes)
    if image is None:
        return []

    image = resize_if_large(image, max_dim=1280)

    try:
        boxes, scores, labels = _detect(image)
    except Exception as e:
        print(f"[object_detection] error: {e}")
        return []

    results = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        results.append({
            "label":       _class_names[label],
            "confidence":       round(score, 4),
            "bbox": {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            }
        })

    return results