import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from arabic_ocr.config import SAUVOLA_WINDOW, MORPH_KERNEL


def binarize(gray):

    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    sauvola_window = max(11, min(SAUVOLA_WINDOW, min(height, width) // 8))  
    win = sauvola_window | 1  


    binary = None
    for k in (0.24, 0.32, 0.40, 0.48):
        thresh = threshold_sauvola(blurred, window_size=win, k=k)
        candidate = (blurred < thresh).astype(np.uint8) * 255
        binary = candidate
        if np.mean(candidate == 0) <= 0.10: 
            break

    if np.mean(binary == 0) > 0.50:
        _, binary = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(binary == 0) > 0.50:
            binary = cv2.bitwise_not(binary)

    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    erode_k = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(opened, erode_k, iterations=1)

    inverted = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        median_area = float(np.median(areas)) if len(areas) > 0 else 3.0
        restore_thresh = max(1, int(0.02 * median_area))
        for lab in range(1, num_labels):
            area = stats[lab, cv2.CC_STAT_AREA]
            if area <= restore_thresh:
                eroded[labels == lab] = 0

    return eroded
