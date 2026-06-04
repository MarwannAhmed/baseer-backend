import cv2
import numpy as np

from app.features.text_extraction.config import parameters


def _rotate(binary, angle):
    h, w = binary.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(binary, M, (w, h),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=255)

def _score(binary, angle):
    rotated = _rotate(binary, angle)
    proj    = np.sum(rotated < 128, axis=1).astype(float)
    return np.var(proj)

def orient(binary):
    if binary.shape[1] > 800:
        new_width = 800
        new_height = int(binary.shape[0] * 800 / binary.shape[1])
        small = cv2.resize(binary, (new_width, new_height), 
                          interpolation=cv2.INTER_NEAREST)
    else:
        small = binary.copy()
    
    coarse_angles = range(-90, 91, parameters["orient"]["course_step"])
    best          = max(coarse_angles, key=lambda a: _score(small, a))

    fine_angles   = np.arange(best - parameters["orient"]["fine_range"], best + parameters["orient"]["fine_range"] + parameters["orient"]["fine_step"], parameters["orient"]["fine_step"])
    best          = float(max(fine_angles, key=lambda a: _score(small, a)))

    if abs(best) < 0.5:
        return binary, 0.0
    
    h, w = binary.shape[:2]
    if max(h, w) >= 32767:
        scale  = 32766 / max(h, w)
        binary = cv2.resize(binary, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_NEAREST)

    corrected = _rotate(binary, best)
    return corrected, best