import cv2
import numpy as np

from app.features.text_extraction.preprocess.orient import orient
from app.features.text_extraction.config import output_paths


def _to_grayscale(image):
    if len(image.shape) == 2:
        return image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def _normalize_scale(gray, target_dpi_height = 1000):
    h = gray.shape[0]
    if h == target_dpi_height:
        return gray
    scale = target_dpi_height / h
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interpolation)

def _denoise(gray):
    return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

def _normalize_background(gray, strength=0.7):
    small = cv2.resize(gray, (gray.shape[1] // 8, gray.shape[0] // 8), interpolation=cv2.INTER_AREA)
    sigma = small.shape[0] // 7
    if sigma % 2 == 0:
        sigma += 1
    bg_small = cv2.GaussianBlur(small, (0, 0), sigma)
    bg = cv2.resize(bg_small, (gray.shape[1], gray.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    bg = np.clip(bg, 1, 255).astype(np.float32)
    norm = cv2.divide(gray.astype(np.float32), bg, scale=255)
    norm = np.clip(norm, 0, 255).astype(np.float32)
    blended  = cv2.addWeighted(norm, strength, gray.astype(np.float32), 1.0 - strength, 0)
    return np.clip(blended, 0, 255).astype(np.uint8)

def _binarize(gray):
    block_size = max(11, (gray.shape[0] // 100) | 1)
    if block_size % 2 == 0:
        block_size += 1
    adaptive = cv2.adaptiveThreshold( gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=block_size, C=15)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined = cv2.bitwise_and(adaptive, otsu)
    return combined

def _morphological_clean(binary):
    base_unit = max(1, binary.shape[0] // 1000)
    noise_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (base_unit, base_unit)
    )
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, noise_kernel, iterations=1)
    base_unit = max(1, binary.shape[0] // 40)
    gap_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (base_unit, base_unit)
    )
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, gap_kernel, iterations=1)
    return closed

def _invert_if_dark_background(binary):
    white_ratio = np.sum(binary == 255) / binary.size
    if white_ratio < 0.5:
        return cv2.bitwise_not(binary)
    return binary

def _separate_characters(binary):
    base_unit = max(1, binary.shape[0] // 500)
    kernel    = cv2.getStructuringElement(
        cv2.MORPH_RECT, (base_unit, 1)
    )
    return cv2.erode(binary, kernel, iterations=1)

def preprocess(image, save_output=False, frame_number=0):
    gray = _to_grayscale(image)
    gray = _normalize_scale(gray)
    gray = _denoise(gray)
    gray = _normalize_background(gray)
    binary = _binarize(gray)
    cleaned = _invert_if_dark_background(binary)
    cleaned = cv2.bitwise_not(cleaned)
    cleaned = _morphological_clean(cleaned)
    cleaned = _separate_characters(cleaned)
    cleaned = cv2.bitwise_not(cleaned)
    oriented, _ = orient(cleaned)
    if save_output:
        cv2.imwrite(f"{output_paths['preprocess']}/denoised_{frame_number}.png",  gray)
        cv2.imwrite(f"{output_paths['preprocess']}/binarized_{frame_number}.png",  binary)
        cv2.imwrite(f"{output_paths['preprocess']}/cleaned_{frame_number}.png",  cleaned)
    return oriented