import cv2
import numpy as np
from skimage.filters import threshold_sauvola

from config import PREPROCESS_DIR
from scripts.english_ocr.attempts.enhance import enhance
from scripts.english_ocr.attempts.filter import filter
from orient import orient


def preprocess(img, frame_number=0):
    gray = enhance(img)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh  = threshold_sauvola(denoised, window_size=25)
    binary  = ((denoised > thresh) * 255).astype(np.uint8)
    kernel  = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # we filter components after orienting to make sure aspect ratios are correct first,
    # logically we should filter before orienting as well to avoid orienting based on noise, 
    # but results show no effect, so removed this filter for now to save computation time
    # filtered = filter_binary(cleaned)

    oriented, angle = orient(cleaned)
    print(f"Orientation correction: {angle:.1f}°")

    filtered = filter(oriented)

    padded = np.pad(filtered, pad_width=int(filtered.shape[0] * 0.2), mode='constant', constant_values=255)

    cv2.imwrite(f"{PREPROCESS_DIR}/denoised_{frame_number}.png",  denoised)
    cv2.imwrite(f"{PREPROCESS_DIR}/binary_{frame_number}.png",    binary)
    cv2.imwrite(f"{PREPROCESS_DIR}/oriented_{frame_number}.png",  oriented)
    cv2.imwrite(f"{PREPROCESS_DIR}/filtered_{frame_number}.png",  filtered)
    cv2.imwrite(f"{PREPROCESS_DIR}/padded_{frame_number}.png",    padded)
    print(f"[1] Preprocessing done")
    return padded