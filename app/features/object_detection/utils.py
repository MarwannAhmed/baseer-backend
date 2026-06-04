"""
app/features/object_detection/utils.py — Detection Helpers
"""

import os
import numpy as np


def compute_iou(box1, box2):
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def non_maximum_suppression(boxes, scores, labels, iou_threshold=0.3):
    if len(boxes) == 0:
        return [], [], []
    boxes  = np.array(boxes,  dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    order  = np.argsort(scores)[::-1]
    keep   = []
    while len(order) > 0:
        best = order[0]
        keep.append(best)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = np.array([compute_iou(boxes[best], boxes[j]) for j in rest])
        order = rest[ious < iou_threshold]
    return boxes[keep].tolist(), scores[keep].tolist(), labels[keep].tolist()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)