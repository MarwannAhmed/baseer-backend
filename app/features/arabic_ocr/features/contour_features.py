import cv2
import numpy as np
from arabic_ocr.config import OUTLINE_SAMPLES 


def contour_features(normalized_img: np.ndarray):
 
    inverted = cv2.bitwise_not(normalized_img) 
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    size = float(normalized_img.shape[0]) 

    if not contours:
        return np.zeros(OUTLINE_SAMPLES * 3, dtype=np.float32)

    contour = max(contours, key=len).reshape(-1, 2).astype(float) 
    n = len(contour)

    if n < 2:
        return np.zeros(OUTLINE_SAMPLES * 3, dtype=np.float32)

    diffs = np.diff(contour, axis=0)
    segment_length = np.hypot(diffs[:, 0], diffs[:, 1])
    cumulative_length = np.concatenate(([0.0], np.cumsum(segment_length)))
    total_length = cumulative_length[-1]

    if total_length == 0:
        return np.zeros(OUTLINE_SAMPLES * 3, dtype=np.float32)

    sample_distance = np.linspace(0, total_length, OUTLINE_SAMPLES, endpoint=False)
    indices = np.searchsorted(cumulative_length, sample_distance, side="right") - 1
    indices = np.clip(indices, 0, n - 1)

    pts = contour[indices]          
    next_idx = (indices + 1) % n
    next_pts = contour[next_idx]

    #computes tangent to see which direction  outline is going at that point, which is important for distinguishing letters with similar shapes but different stroke directions (e.g. ر vs و)
    tangent_angles = np.arctan2(  
        next_pts[:, 1] - pts[:, 1],
        next_pts[:, 0] - pts[:, 0],
    )
    tangent_normalized = (tangent_angles + np.pi) / (2 * np.pi)  

    feat = np.stack([
        pts[:, 0] / size,
        pts[:, 1] / size,
        tangent_normalized,
    ], axis=1).reshape(-1).astype(np.float32)

    return feat
