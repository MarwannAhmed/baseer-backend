import cv2
import numpy as np

MIN_AREA_FRACTION = 0.1
MIN_WH_RATIO = 0.1
MAX_WH_RATIO = 10.0


def filter(binary):
    inv = 255 - binary
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)

    areas = [stats[i][4] for i in range(1, n)]
    min_area = np.median(areas) * MIN_AREA_FRACTION if areas else 1

    mask = np.zeros_like(inv)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        ratio = w / (h + 1e-5)
        if not (MIN_WH_RATIO < ratio < MAX_WH_RATIO):
            continue
        mask[labels == i] = 255

    return 255 - mask