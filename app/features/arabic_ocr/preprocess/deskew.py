import cv2
import numpy as np
from arabic_ocr.config import SKEW_ANGLE_MAX


def deskew(binary):
    height, width = binary.shape

    # Strip outer 2% of the image to exclude scanner border artefacts
    margin_height = max(1, int(height * 0.02))
    margin_width = max(1, int(width * 0.02))
    interior = binary[margin_height: height - margin_height, margin_width: width - margin_width]

    angle = _projection_skew(interior)
    if angle is None:
        angle = _bbox_skew(interior)
    if angle is None:
        return binary.copy(), 0.0

    center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0) #to rotate around center point
    rotated = cv2.warpAffine(
        binary, M, (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255, #fill new corners with white
    )
    return rotated, float(angle)


def _projection_skew(binary, angle_range = 15.0):

    height, width = binary.shape
    if np.sum(binary == 0) < 50:
        return None

    center = (width / 2, height / 2)
    best_angle = 0.0
    best_var = _row_proj_var(binary)
    baseline_var = best_var

    for deg_10 in range(int(-angle_range * 10), int(angle_range * 10) + 1, 5):
        angle = deg_10 / 10.0
        if angle == 0.0:
            continue
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            binary, M, (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        var = _row_proj_var(rotated)
        if var > best_var:
            best_var = var
            best_angle = angle

    # Only apply correction when variance improvement is meaningful (>5%) (this is to avoid microcorrection on straight imgs)
    if best_var < baseline_var * 1.05:
        return 0.0
    if abs(best_angle) > SKEW_ANGLE_MAX:
        return None
    return best_angle


def _row_proj_var(binary):
    _projection_skew = np.sum(binary == 0, axis=1).astype(float)
    return float(np.var(_projection_skew))


def _bbox_skew(binary):
    coords = np.column_stack(np.where(binary == 0))
    if len(coords) < 10:
        return None

    rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
    angle = rect[-1]
    if angle < -45:
        angle += 90
    if abs(angle) > SKEW_ANGLE_MAX:
        return None
    return float(angle)
