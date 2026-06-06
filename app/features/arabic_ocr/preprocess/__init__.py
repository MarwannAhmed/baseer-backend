import cv2
import numpy as np
import arabic_ocr.config as _cfg
from .enhance import enhance
from .binarize import binarize
from .deskew import deskew
from .filter import filter_noise

_MIN_HEIGHT = 300  

def _upscale(img):

    height = img.shape[0]
    if height >= _MIN_HEIGHT:
        return img
    scale = _MIN_HEIGHT / height
    new_w = int(img.shape[1] * scale)
    return cv2.resize(img, (new_w, _MIN_HEIGHT), interpolation=cv2.INTER_CUBIC)

def preprocess(img, frame_number= 0):
    img = _upscale(img)
    gray = enhance(img)
    binary = binarize(gray)
    clean = filter_noise(binary)
    deskewed, angle = deskew(clean)

    height, width = deskewed.shape
    pad_height = int(height * 0.05)
    pad_width = int(width * 0.05)
    padded = np.pad(deskewed, ((pad_height, pad_height), (pad_width, pad_width)), constant_values=255)

    if _cfg.DEBUG:
        _cfg.PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(_cfg.PREPROCESS_DIR / f"{frame_number:04d}_gray.png"),   gray)
        cv2.imwrite(str(_cfg.PREPROCESS_DIR / f"{frame_number:04d}_binary.png"), binary)
        cv2.imwrite(str(_cfg.PREPROCESS_DIR / f"{frame_number:04d}_clean.png"),  clean)
        cv2.imwrite(str(_cfg.PREPROCESS_DIR / f"{frame_number:04d}_deskew.png"), deskewed)
        cv2.imwrite(str(_cfg.PREPROCESS_DIR / f"{frame_number:04d}_padded.png"), padded)
    return padded

__all__ = ["preprocess"]
