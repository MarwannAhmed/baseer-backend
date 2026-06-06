from pathlib import Path
import cv2
import numpy as np

def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def save_image(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True) 
    cv2.imwrite(str(path), img)

def resize_if_large(img, max_dim = 2000):
    height, width = img.shape[:2]
    larger = max(height, width)
    if larger <= max_dim:
        return img
    scale = max_dim / larger
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
