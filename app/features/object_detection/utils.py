import numpy as np


def compute_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    intersection_width = max(0, x_right - x_left)
    intersection_height = max(0, y_bottom - y_top)
    intersection_area = intersection_width * intersection_height

    if intersection_area == 0:
        return 0.0

    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area_box1 + area_box2 - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def non_maximum_suppression(boxes, scores, labels, iou_threshold=0.3):
    if not boxes:
        return [], [], []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    sorted_indices = np.argsort(scores)[::-1]
    kept_indices = []

    while len(sorted_indices) > 0:
        best_index = sorted_indices[0]
        kept_indices.append(best_index)

        if len(sorted_indices) == 1:
            break

        remaining_indices = sorted_indices[1:]

        ious = np.array(
            [
                compute_iou(boxes[best_index], boxes[index])
                for index in remaining_indices
            ]
        )

        sorted_indices = remaining_indices[ious < iou_threshold]

    return (
        boxes[kept_indices].tolist(),
        scores[kept_indices].tolist(),
        labels[kept_indices].tolist(),
    )
