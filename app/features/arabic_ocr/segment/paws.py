import cv2
import numpy as np


def segment_paws(line_binary,ah= 20.0):
    height, width = line_binary.shape

    dilate_width = max(1, int(round(0.25 * ah)))  # Dilates proportionally to average character height so intra-word gaps are closed consistently  

    # Dilate text blobs horizontally to close intra-word gaps.
    kernel = np.ones((1, dilate_width), np.uint8)
    text_white = cv2.bitwise_not(line_binary)
    merged_text = cv2.dilate(text_white, kernel)
    merged = cv2.bitwise_not(merged_text)  # back to text=0, bg=255

    column_proj = np.sum(merged == 0, axis=0)
    candidates = _group_nonzero(column_proj)   # zero columns = word gaps
    min_paw_w = max(1, int(round(0.3 * ah)))
    candidates = [(x1, x2) for x1, x2 in candidates if (x2 - x1) >= min_paw_w]
    if not candidates:
        return [(0, width, line_binary)]

    result = []
    for px1, px2 in candidates:
        crop = line_binary[:, px1:px2]
        if crop.shape[1] > 0:
            result.append((px1, px2, crop))

    if not result:
        return [(0, width, line_binary)]

    # Sort right to left (Arabic reading order)
    result.sort(key=lambda t: t[0], reverse=True)
    return result


def _group_nonzero(proj: np.ndarray) -> list[tuple[int, int]]:
    groups = []
    in_group = False
    start = 0
    for i, v in enumerate(proj):
        if v > 0 and not in_group:
            start = i
            in_group = True
        elif v == 0 and in_group:
            groups.append((start, i))
            in_group = False
    if in_group:
        groups.append((start, len(proj)))
    return groups
