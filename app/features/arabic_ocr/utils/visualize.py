from pathlib import Path
from typing import Sequence
import cv2
import numpy as np


def _thickness(img):
    return 1 if img.shape[0] < 300 else 2

def draw_lines(img, line_bounds):
    out = _ensure_bgr(img)
    t = _thickness(out)
    for y1, y2 in line_bounds:
        cv2.rectangle(out, (0, y1), (out.shape[1] - 1, y2), (0, 220, 0), t)
    return out

def draw_paws(img, paw_boxes):
    out = _ensure_bgr(img)
    t = _thickness(out)
    for idx, (x1, y1, x2, y2) in enumerate(paw_boxes):
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 100, 0), t)
        cv2.putText(out, str(idx), (max(0, x1 + 1), max(8, y1 + 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 0, 200), 1) #writes index in box top left
    return out

def draw_chars(img, character_boxes):
    out = _ensure_bgr(img)
    t = _thickness(out)
    for x1, y1, x2, y2 in character_boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 220), t)
    return out

def draw_dots(img, dot_list):
    out = _ensure_bgr(img)
    for dot in dot_list:
        cx, cy = int(dot.cx), int(dot.cy)
        cv2.circle(out, (cx, cy), max(2, out.shape[0] // 60), (0, 220, 220), -1) #dot radius scales with height so dots are visible
    return out

def save_debug_visualization(img, stage_name, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{stage_name}.png"), img)

def _ensure_bgr(img):
    if img.ndim == 2: #check grayscale
        return cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    return img.copy()
