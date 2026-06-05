import cv2
import numpy as np


def filter_noise(binary):
    inverted = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    if num_labels <= 1:
        return binary.copy()

    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background label 0
    if len(areas) == 0:
        return binary.copy()

    median_area = float(np.median(areas))
    area_threshold = max(3.0, 0.01 * median_area)

    cleaned = np.zeros_like(inverted)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]

        if area < area_threshold:
            continue
        aspect = max(width, height) / max(min(width, height), 1)
        if aspect > 25:
            continue

        cleaned[labels == label] = 255

    return cv2.bitwise_not(cleaned)
