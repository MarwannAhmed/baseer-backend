from pathlib import Path
import cv2
import numpy as np


def crop_region(img, x1, y1, x2, y2, padding: int = 2):
    height, width = img.shape[:2]
    y1_clip = max(0, y1 - padding)
    y2_clip = min(height, y2 + padding)
    x1_clip = max(0, x1 - padding)
    x2_clip = min(width, x2 + padding)
    return img[y1_clip:y2_clip, x1_clip:x2_clip]


def save_crops(crops, directory, prefix = "crop"):
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, crop in enumerate(crops):
        cv2.imwrite(str(out_dir / f"{prefix}_{idx:04d}.png"), crop)
